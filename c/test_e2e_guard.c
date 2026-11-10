/* Unit tests of e2e_guard (run by `make test` in CI). Each test names the DOSSIER requirement it verifies. */
#include "e2e_guard.h"

#include <stdio.h>
#include <string.h>

static int failures = 0;
static int checks = 0;

#define CHECK(cond)                                                                    \
    do {                                                                               \
        checks++;                                                                      \
        if (!(cond)) {                                                                 \
            failures++;                                                                \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                     \
        }                                                                              \
    } while (0)

static void make(e2e_tx_t *tx, uint8_t *pdu, uint8_t angle)
{
    memset(pdu, 0, 8);
    pdu[2] = angle;
    pdu[3] = (uint8_t)(angle ^ 0x5Au);
    e2e_protect(tx, pdu, 8);
}

/* SR-52: CRC-8 is the SAE J1850 polynomial with 0x00 start/xor (check value of "123456789" = 0x37) */
static void test_crc(void)
{
    const uint8_t s[] = "123456789";
    CHECK(e2e_crc8(s, 9, 0x00) == 0x37);
    CHECK((uint8_t)(e2e_crc8(s, 9, 0xFF) ^ 0xFF) == 0x4B); /* the 0xFF/0xFF variant */
}

/* SR-52: counter sequence 0..14 skips 0xF; every corruption of one bit is caught */
static void test_protect_check(void)
{
    e2e_tx_t tx;
    e2e_rx_t rx;
    uint8_t pdu[8];
    e2e_tx_init(&tx, 0x1234);
    e2e_rx_init(&rx, 0x1234, 2);
    make(&tx, pdu, 10);
    CHECK(e2e_check(&rx, pdu, 8) == E2E_INITIAL);
    for (int i = 1; i < 40; i++) {
        make(&tx, pdu, (uint8_t)i);
        CHECK((pdu[1] & 0x0F) != 0x0F);
        CHECK(e2e_check(&rx, pdu, 8) == E2E_OK);
    }
    int caught = 0;
    for (int byte = 0; byte < 8; byte++) {
        for (int bit = 0; bit < 8; bit++) {
            e2e_rx_t r2 = rx;
            uint8_t bad[8];
            make(&tx, pdu, 77);
            memcpy(bad, pdu, 8);
            bad[byte] ^= (uint8_t)(1u << bit);
            caught += e2e_check(&r2, bad, 8) == E2E_ERROR;
            e2e_check(&rx, pdu, 8);
        }
    }
    CHECK(caught == 64);
}

/* SR-52: masquerade: a correct frame checked under another data ID fails */
static void test_masquerade(void)
{
    e2e_tx_t tx;
    e2e_rx_t rx;
    uint8_t pdu[8];
    e2e_tx_init(&tx, 0x0101);
    e2e_rx_init(&rx, 0x0102, 2);
    make(&tx, pdu, 5);
    CHECK(e2e_check(&rx, pdu, 8) == E2E_ERROR);
}

/* SR-52: repetition, loss within delta, and a sequence jump are classified */
static void test_sequence(void)
{
    e2e_tx_t tx;
    e2e_rx_t rx;
    uint8_t p0[8], p1[8], p2[8], p3[8];
    e2e_tx_init(&tx, 7);
    e2e_rx_init(&rx, 7, 2);
    make(&tx, p0, 1);
    make(&tx, p1, 2);
    make(&tx, p2, 3);
    make(&tx, p3, 4);
    CHECK(e2e_check(&rx, p0, 8) == E2E_INITIAL);
    CHECK(e2e_check(&rx, p0, 8) == E2E_REPEATED);
    CHECK(e2e_check(&rx, p2, 8) == E2E_OKSOMELOST);
    for (int i = 0; i < 5; i++) {
        make(&tx, p0, 9);
    }
    CHECK(e2e_check(&rx, p0, 8) == E2E_WRONGSEQUENCE);
}

/* SR-50, SR-51: losing channel A leaves the value fresh through B; losing both is detected within the timeout */
static void test_redundancy(void)
{
    e2e_tx_t tx;
    rm_t rm;
    uint8_t pdu[8];
    e2e_tx_init(&tx, 0x0EF0);
    rm_init(&rm, 0x0EF0, 30, 2);
    uint32_t t = 0;
    uint32_t worst_age = 0;
    for (int k = 0; k < 300; k++, t += 10) {
        make(&tx, pdu, (uint8_t)k);
        if (k < 100) {
            rm_on_frame(&rm, 0, pdu, 8, t + 1);
        }
        if (k < 200) {
            rm_on_frame(&rm, 1, pdu, 8, t + 2);
        }
        rm_state_t s = rm_tick(&rm, t + 5);
        if (k > 5 && k < 100) {
            CHECK(s == RM_DUAL);
        }
        if (k >= 104 && k < 200) {
            CHECK(s == RM_SINGLE_B);
            uint32_t age = rm_value_age(&rm, t + 5);
            if (age > worst_age) {
                worst_age = age;
            }
        }
        if (k >= 204) {
            CHECK(s == RM_LOST);
        }
    }
    CHECK(worst_age <= 10);
    CHECK(rm.duplicates >= 99);
    CHECK(rm.disagree == 0);
}

/* SR-52: a copy that differs from its twin with the same counter is flagged */
static void test_disagree(void)
{
    e2e_tx_t ta, tb;
    rm_t rm;
    uint8_t a[8], b[8];
    e2e_tx_init(&ta, 3);
    e2e_tx_init(&tb, 3);
    rm_init(&rm, 3, 30, 2);
    make(&ta, a, 1);
    make(&tb, b, 2); /* same counter, different angle: a faulty sender on one channel */
    rm_on_frame(&rm, 0, a, 8, 0);
    rm_on_frame(&rm, 1, b, 8, 1);
    CHECK(rm.disagree == 1);
}

/* SR-53: the guardian passes the configured rate and latches a babbling transmitter off */
static void test_guardian(void)
{
    guard_t g;
    guard_init(&g, 100, 2, 20, 0);
    int sent = 0;
    for (uint32_t t = 0; t < 1000; t += 10) {
        sent += guard_allow(&g, t);
    }
    CHECK(sent >= 99 && sent <= 102);
    CHECK(!g.tripped);
    int babble = 0;
    for (uint32_t i = 0; i < 1000; i++) {
        babble += guard_allow(&g, 1000);
    }
    CHECK(babble <= 2);
    CHECK(g.tripped);
}

int main(void)
{
    test_crc();
    test_protect_check();
    test_masquerade();
    test_sequence();
    test_redundancy();
    test_disagree();
    test_guardian();
    printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
