/*
 * e2e_guard: end-to-end protection, dual-channel redundancy management and a transmit bus guardian
 * for the by-wire CAN frames of the vetronics network study (DOSSIER section 9).
 *
 * E2E header in the style of the AUTOSAR E2E Profile 1 (counter, timeout via counter, data ID, CRC):
 *   byte 0      CRC-8, polynomial 0x1D (SAE J1850), start value 0x00, final XOR 0x00, computed over
 *               data-ID low byte, data-ID high byte, then bytes 1..len-1 of the PDU;
 *   byte 1 b3-0 4-bit counter 0..14 (0xF is never sent), incremented on every send.
 * The 16-bit data ID is not transmitted; it is folded into the CRC, so a frame received under the wrong
 * identifier (masquerade, wrong routing) fails the CRC.
 *
 * Plain C99, no dynamic memory, no floating point: suitable for a safety-related ECU.
 */
#ifndef E2E_GUARD_H
#define E2E_GUARD_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define E2E_COUNTER_MOD 15u

uint8_t e2e_crc8(const uint8_t *data, unsigned len, uint8_t start);

typedef struct {
    uint16_t data_id;
    uint8_t counter;
} e2e_tx_t;

void e2e_tx_init(e2e_tx_t *s, uint16_t data_id);
/* Writes the counter into pdu[1] (low nibble) and the CRC into pdu[0]; len >= 2. */
void e2e_protect(e2e_tx_t *s, uint8_t *pdu, unsigned len);

typedef enum {
    E2E_OK = 0,            /* new data, counter incremented by 1 */
    E2E_OKSOMELOST = 1,    /* new data, counter jumped by 2..max_delta */
    E2E_REPEATED = 2,      /* same counter as the last accepted frame */
    E2E_WRONGSEQUENCE = 3, /* counter jumped by more than max_delta */
    E2E_ERROR = 4,         /* CRC, length or counter-range error */
    E2E_INITIAL = 5        /* first valid frame after start */
} e2e_status_t;

typedef struct {
    uint16_t data_id;
    uint8_t max_delta;
    uint8_t last;
    bool synced;
} e2e_rx_t;

void e2e_rx_init(e2e_rx_t *s, uint16_t data_id, uint8_t max_delta);
e2e_status_t e2e_check(e2e_rx_t *s, const uint8_t *pdu, unsigned len);

/* ---- dual-channel redundancy manager ---------------------------------------------------------- */
typedef enum { RM_INIT = 0, RM_DUAL = 1, RM_SINGLE_A = 2, RM_SINGLE_B = 3, RM_LOST = 4 } rm_state_t;

#define RM_MAX_LEN 8u

typedef struct {
    e2e_rx_t rx[2];
    uint32_t last_ok_ms[2];
    bool seen[2];
    uint32_t timeout_ms;
    uint8_t value[RM_MAX_LEN];
    uint8_t len;
    uint8_t value_counter;
    bool have_value;
    uint32_t value_ms;
    rm_state_t state;
    uint32_t accepted;
    uint32_t duplicates;
    uint32_t disagree;
    uint32_t rejected[2];
} rm_t;

void rm_init(rm_t *rm, uint16_t data_id, uint32_t timeout_ms, uint8_t max_delta);
/* Feed a frame received on channel ch (0 = A, 1 = B) at time now_ms; returns the E2E status of that copy. */
e2e_status_t rm_on_frame(rm_t *rm, unsigned ch, const uint8_t *pdu, unsigned len, uint32_t now_ms);
/* Update the channel-health state at time now_ms. */
rm_state_t rm_tick(rm_t *rm, uint32_t now_ms);
/* Age of the value the application uses, ms (UINT32_MAX if none yet). */
uint32_t rm_value_age(const rm_t *rm, uint32_t now_ms);

/* ---- transmit bus guardian (babbling-idiot protection) ---------------------------------------- */
typedef struct {
    uint32_t tokens_milli; /* available frames x 1000 */
    uint32_t depth_milli;
    uint32_t rate_per_s;   /* allowed frames per second */
    uint32_t last_ms;
    uint32_t violations;
    uint32_t trip_after;   /* consecutive violations that latch the transmitter off (0 = never) */
    uint32_t consecutive;
    bool tripped;
} guard_t;

void guard_init(guard_t *g, uint32_t rate_per_s, uint32_t depth_frames, uint32_t trip_after, uint32_t now_ms);
/* True if the frame may go on the bus; false = dropped (and the guard may latch off). */
bool guard_allow(guard_t *g, uint32_t now_ms);

#ifdef __cplusplus
}
#endif
#endif
