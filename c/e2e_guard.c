#include "e2e_guard.h"

#include <stddef.h>
#include <string.h>

uint8_t e2e_crc8(const uint8_t *data, unsigned len, uint8_t start)
{
    uint8_t crc = start;
    for (unsigned i = 0; i < len; i++) {
        crc ^= data[i];
        for (int b = 0; b < 8; b++) {
            crc = (crc & 0x80u) ? (uint8_t)(((unsigned)crc << 1u) ^ 0x1Du) : (uint8_t)((unsigned)crc << 1u);
        }
    }
    return crc;
}

static uint8_t pdu_crc(uint16_t data_id, const uint8_t *pdu, unsigned len)
{
    uint8_t id[2] = {(uint8_t)(data_id & 0xFFu), (uint8_t)(data_id >> 8)};
    uint8_t crc = e2e_crc8(id, 2u, 0x00u);
    return e2e_crc8(pdu + 1, len - 1u, crc);
}

void e2e_tx_init(e2e_tx_t *s, uint16_t data_id)
{
    s->data_id = data_id;
    s->counter = 0u;
}

void e2e_protect(e2e_tx_t *s, uint8_t *pdu, unsigned len)
{
    if (len < 2u) {
        return;
    }
    pdu[1] = (uint8_t)((pdu[1] & 0xF0u) | (s->counter & 0x0Fu));
    pdu[0] = pdu_crc(s->data_id, pdu, len);
    s->counter = (uint8_t)((s->counter + 1u) % E2E_COUNTER_MOD);
}

void e2e_rx_init(e2e_rx_t *s, uint16_t data_id, uint8_t max_delta)
{
    s->data_id = data_id;
    s->max_delta = max_delta;
    s->last = 0u;
    s->synced = false;
}

e2e_status_t e2e_check(e2e_rx_t *s, const uint8_t *pdu, unsigned len)
{
    if (pdu == NULL || len < 2u || len > 64u) {
        return E2E_ERROR;
    }
    uint8_t c = (uint8_t)(pdu[1] & 0x0Fu);
    if (c >= E2E_COUNTER_MOD || pdu[0] != pdu_crc(s->data_id, pdu, len)) {
        return E2E_ERROR;
    }
    if (!s->synced) {
        s->synced = true;
        s->last = c;
        return E2E_INITIAL;
    }
    uint8_t delta = (uint8_t)((c + E2E_COUNTER_MOD - s->last) % E2E_COUNTER_MOD);
    if (delta == 0u) {
        return E2E_REPEATED;
    }
    s->last = c; /* on a jump the receiver resynchronises to the new sequence but reports it */
    if (delta > s->max_delta) {
        return E2E_WRONGSEQUENCE;
    }
    return delta == 1u ? E2E_OK : E2E_OKSOMELOST;
}

void rm_init(rm_t *rm, uint16_t data_id, uint32_t timeout_ms, uint8_t max_delta)
{
    memset(rm, 0, sizeof(*rm));
    e2e_rx_init(&rm->rx[0], data_id, max_delta);
    e2e_rx_init(&rm->rx[1], data_id, max_delta);
    rm->timeout_ms = timeout_ms;
    rm->state = RM_INIT;
}

/* counter a is newer than b if it lies 1..7 steps ahead modulo 15 */
static bool newer(uint8_t a, uint8_t b)
{
    uint8_t d = (uint8_t)((a + E2E_COUNTER_MOD - b) % E2E_COUNTER_MOD);
    return d >= 1u && d <= 7u;
}

e2e_status_t rm_on_frame(rm_t *rm, unsigned ch, const uint8_t *pdu, unsigned len, uint32_t now_ms)
{
    if (ch > 1u || len > RM_MAX_LEN) {
        return E2E_ERROR;
    }
    e2e_status_t st = e2e_check(&rm->rx[ch], pdu, len);
    if (st == E2E_ERROR || st == E2E_REPEATED) {
        rm->rejected[ch]++;
        return st;
    }
    /* a valid copy refreshes the health of its channel */
    rm->last_ok_ms[ch] = now_ms;
    rm->seen[ch] = true;
    uint8_t c = (uint8_t)(pdu[1] & 0x0Fu);
    if (rm->have_value && c == rm->value_counter) {
        rm->duplicates++;
        if (len != rm->len || memcmp(pdu, rm->value, len) != 0) {
            rm->disagree++; /* same counter, different content: one copy is corrupt beyond its CRC */
        }
        return st;
    }
    if (!rm->have_value || newer(c, rm->value_counter) || st == E2E_WRONGSEQUENCE ||
        (uint32_t)(now_ms - rm->value_ms) > rm->timeout_ms) {
        memcpy(rm->value, pdu, len);
        rm->len = (uint8_t)len;
        rm->value_counter = c;
        rm->value_ms = now_ms;
        rm->have_value = true;
        rm->accepted++;
    }
    return st;
}

rm_state_t rm_tick(rm_t *rm, uint32_t now_ms)
{
    bool ok[2];
    for (unsigned i = 0; i < 2u; i++) {
        ok[i] = rm->seen[i] && (uint32_t)(now_ms - rm->last_ok_ms[i]) <= rm->timeout_ms;
    }
    if (!rm->seen[0] && !rm->seen[1]) {
        rm->state = RM_INIT;
    } else if (ok[0] && ok[1]) {
        rm->state = RM_DUAL;
    } else if (ok[0]) {
        rm->state = RM_SINGLE_A;
    } else if (ok[1]) {
        rm->state = RM_SINGLE_B;
    } else {
        rm->state = RM_LOST;
    }
    return rm->state;
}

uint32_t rm_value_age(const rm_t *rm, uint32_t now_ms)
{
    return rm->have_value ? (uint32_t)(now_ms - rm->value_ms) : UINT32_MAX;
}

void guard_init(guard_t *g, uint32_t rate_per_s, uint32_t depth_frames, uint32_t trip_after, uint32_t now_ms)
{
    g->rate_per_s = rate_per_s;
    g->depth_milli = depth_frames * 1000u;
    g->tokens_milli = g->depth_milli;
    g->last_ms = now_ms;
    g->violations = 0u;
    g->trip_after = trip_after;
    g->consecutive = 0u;
    g->tripped = false;
}

bool guard_allow(guard_t *g, uint32_t now_ms)
{
    if (g->tripped) {
        return false;
    }
    uint32_t dt = (uint32_t)(now_ms - g->last_ms);
    g->last_ms = now_ms;
    uint64_t t = (uint64_t)g->tokens_milli + (uint64_t)dt * g->rate_per_s; /* frames x 1000 accrued in dt ms */
    g->tokens_milli = (uint32_t)(t > g->depth_milli ? g->depth_milli : t);
    if (g->tokens_milli >= 1000u) {
        g->tokens_milli -= 1000u;
        g->consecutive = 0u;
        return true;
    }
    g->violations++;
    g->consecutive++;
    if (g->trip_after != 0u && g->consecutive >= g->trip_after) {
        g->tripped = true; /* latch the transmitter off: a babbling node must not take the bus down */
    }
    return false;
}
