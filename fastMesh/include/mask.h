#ifndef MASK_H__
#define MASK_H__

/// @return Number of bits that are on in the specified 64-bit word
__hostdev__ inline uint32_t CountOn(uint64_t v)
{
#if (defined(__CUDA_ARCH__) || defined(__HIP__)) && defined(NANOVDB_USE_INTRINSICS)
//#warning Using popcll for CountOn
    return __popcll(v);
// __popcnt64 intrinsic support was added in VS 2019 16.8
#elif defined(_MSC_VER) && defined(_M_X64) && (_MSC_VER >= 1928) && defined(NANOVDB_USE_INTRINSICS)
//#warning Using popcnt64 for CountOn
    return __popcnt64(v);
#elif (defined(__GNUC__) || defined(__clang__)) && defined(NANOVDB_USE_INTRINSICS)
//#warning Using builtin_popcountll for CountOn
    return __builtin_popcountll(v);
#else// use software implementation
//#warning Using software implementation for CountOn
    v = v - ((v >> 1) & uint64_t(0x5555555555555555));
    v = (v & uint64_t(0x3333333333333333)) + ((v >> 2) & uint64_t(0x3333333333333333));
    return (((v + (v >> 4)) & uint64_t(0xF0F0F0F0F0F0F0F)) * uint64_t(0x101010101010101)) >> 56;
#endif
}

__hostdev__ static inline uint32_t FindLowestOn(uint64_t v)
{
#if (defined(__CUDA_ARCH__) || defined(__HIP__)) && defined(NANOVDB_USE_INTRINSICS)
    return __ffsll(v);
#elif defined(_MSC_VER) && defined(NANOVDB_USE_INTRINSICS)
    unsigned long index;
    _BitScanForward64(&index, v);
    return static_cast<uint32_t>(index);
#elif (defined(__GNUC__) || defined(__clang__)) && defined(NANOVDB_USE_INTRINSICS)
    return static_cast<uint32_t>(__builtin_ctzll(v));
#else
//#warning Using software implementation for FindLowestOn(uint64_t)
    static const unsigned char DeBruijn[64] = {
        0,   1,  2, 53,  3,  7, 54, 27, 4,  38, 41,  8, 34, 55, 48, 28,
        62,  5, 39, 46, 44, 42, 22,  9, 24, 35, 59, 56, 49, 18, 29, 11,
        63, 52,  6, 26, 37, 40, 33, 47, 61, 45, 43, 21, 23, 58, 17, 10,
        51, 25, 36, 32, 60, 20, 57, 16, 50, 31, 19, 15, 30, 14, 13, 12,
    };
// disable unary minus on unsigned warning
#if defined(_MSC_VER) && !defined(__NVCC__)
#pragma warning(push)
#pragma warning(disable : 4146)
#endif
    return DeBruijn[uint64_t((v & -v) * UINT64_C(0x022FDD63CC95386D)) >> 58];
#if defined(_MSC_VER) && !defined(__NVCC__)
#pragma warning(pop)
#endif

#endif
}

/*
Code from NanoVDB
*/
template<uint32_t Size>
class Mask
{
    static constexpr uint32_t SIZE = Size; // Number of bits in mask
    static constexpr uint32_t WORD_COUNT = SIZE >> 6; // Number of 64 bit words
    uint64_t mWords[WORD_COUNT];

public:
    /// @brief Return the memory footprint in bytes of this Mask
    __hostdev__ static size_t memUsage() { return sizeof(Mask); }

    /// @brief Initialize all bits to zero.
    __hostdev__ Mask()
    {
        for (uint32_t i = 0; i < WORD_COUNT; ++i)
            mWords[i] = 0;
    }

    __hostdev__ Mask(uint64_t* _mWords)
    {
        for (uint32_t i = 0; i < WORD_COUNT; ++i)
            mWords[i] = _mWords[i];
    }

    __hostdev__ void reset(){
        for (uint32_t i = 0; i < WORD_COUNT; ++i)
            mWords[i] = 0;
    }

    /// @brief Return true if the given bit is set.
    __hostdev__ bool isOn(uint32_t n) const { return 0 != (mWords[n >> 6] & (uint64_t(1) << (n & 63))); }

    /// @brief Return true if the given bit is NOT set.
    __hostdev__ bool isOff(uint32_t n) const { return 0 == (mWords[n >> 6] & (uint64_t(1) << (n & 63))); }

    /// @brief Set the specified bit on.
    __hostdev__ void setOn(uint32_t n) { mWords[n >> 6] |= uint64_t(1) << (n & 63); }

    /// @brief Set the specified bit off.
    __hostdev__ void setOff(uint32_t n) { mWords[n >> 6] &= ~(uint64_t(1) << (n & 63)); }

    /// @brief Return the total number of set bits in this Mask
    __hostdev__ uint32_t countOn() const
    {
        uint32_t sum = 0, n = WORD_COUNT;
        for (const uint64_t* w = mWords; n--; ++w)
            sum += CountOn(*w);
        return sum;
    }

    /// @brief Return the number of lower set bits in mask up to but excluding the i'th bit
    inline __hostdev__ uint32_t countOn(uint32_t i) const
    {
        uint32_t n = i >> 6, sum = CountOn( mWords[n] & ((uint64_t(1) << (i & 63u))-1u) );
        for (const uint64_t* w = mWords; n--; ++w) sum += CountOn(*w);
        return sum;
    }

};

#endif 