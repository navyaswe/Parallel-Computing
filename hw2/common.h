#ifndef COMMON_H_
#define COMMON_H_

#include <stdint.h>
#include <stdlib.h>

#define CACHELINE      64
#define PAGESIZE       4096

static inline uint64_t ReadTSC(void)
{
#if defined(__i386__)

    uint64_t x;
    __asm__ __volatile__(".byte 0x0f, 0x31":"=A"(x));
    return x;

#elif defined(__x86_64__)

    uint32_t hi, lo;
    __asm__ __volatile__("rdtsc":"=a"(lo), "=d"(hi));
    return ((uint64_t) lo) | (((uint64_t) hi) << 32);

#elif defined(__powerpc__)

    uint64_t result = 0;
    uint64_t upper, lower, tmp;
    __asm__ __volatile__("0:                  \n"
                         "\tmftbu   %0           \n"
                         "\tmftb    %1           \n"
                         "\tmftbu   %2           \n"
                         "\tcmpw    %2,%0        \n"
                         "\tbne     0b         \n":"=r"(upper), "=r"(lower),
                         "=r"(tmp)
        );
    result = upper;
    result = result << 32;
    result = result | lower;
    return result;

#endif // defined(__i386__)
}

static inline size_t AlignedSize(size_t size)
{
    size_t aligned_size = (size + PAGESIZE - 1)/PAGESIZE * PAGESIZE + PAGESIZE;
    return aligned_size;
}


static inline void *AlignedMalloc(size_t size)
{
    void *addr = NULL;
    size_t aligned_size = AlignedSize(size);
    if (posix_memalign(&addr, PAGESIZE, aligned_size) != 0) {
        addr = NULL;
    }
    return addr;
}

static inline void AlignedFree(void *addr)
{
    if (addr != NULL) {
        free(addr);
    }
    addr = NULL;
}


void InitTSC(void);

double ElapsedTime(uint64_t ticks);


#endif // COMMON_H_

