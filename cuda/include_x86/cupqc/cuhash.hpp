// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUHASH_HPP
#define CUHASH_HPP

#include "cupqc.hpp"

namespace cupqc {
    using SHA2_224    = decltype(Algorithm<algorithm::SHA2_32>() + SecurityCategory<1>() + Function<function::Hash>());
    using SHA2_256    = decltype(Algorithm<algorithm::SHA2_32>() + SecurityCategory<2>() + Function<function::Hash>());
    
    using SHA2_512_224  = decltype(Algorithm<algorithm::SHA2_64>() + SecurityCategory<1>() + Function<function::Hash>());
    using SHA2_512_256  = decltype(Algorithm<algorithm::SHA2_64>() + SecurityCategory<2>() + Function<function::Hash>());
    using SHA2_384      = decltype(Algorithm<algorithm::SHA2_64>() + SecurityCategory<4>() + Function<function::Hash>());
    using SHA2_512      = decltype(Algorithm<algorithm::SHA2_64>() + SecurityCategory<5>() + Function<function::Hash>());

    using SHA3_224    = decltype(Algorithm<algorithm::SHA3>()   + SecurityCategory<1>() + Function<function::Hash>());
    using SHA3_256    = decltype(Algorithm<algorithm::SHA3>()   + SecurityCategory<2>() + Function<function::Hash>());
    using SHA3_384    = decltype(Algorithm<algorithm::SHA3>()   + SecurityCategory<4>() + Function<function::Hash>());
    using SHA3_512    = decltype(Algorithm<algorithm::SHA3>()   + SecurityCategory<5>() + Function<function::Hash>());
    using SHAKE_128   = decltype(Algorithm<algorithm::SHAKE>()  + SecurityCategory<1>() + Function<function::Hash>());
    using SHAKE_256   = decltype(Algorithm<algorithm::SHAKE>()  + SecurityCategory<2>() + Function<function::Hash>());

    using POSEIDON2_8_16      = decltype(Algorithm<algorithm::POSEIDON2>()  + Capacity<8>()  + Width<16>()  + Field<field::BabyBear>() + Function<function::Hash>());
    using POSEIDON2_8_24      = decltype(Algorithm<algorithm::POSEIDON2>()  + Capacity<8>()  + Width<24>()  + Field<field::BabyBear>() + Function<function::Hash>());
    using POSEIDON2_8_16_Test = decltype(Algorithm<algorithm::POSEIDON2>()  + Capacity<8>()  + Width<16>()  + Field<field::BabyBearDefault>() + Function<function::Hash>());
    using POSEIDON2_8_24_Test = decltype(Algorithm<algorithm::POSEIDON2>()  + Capacity<8>()  + Width<24>()  + Field<field::BabyBearDefault>() + Function<function::Hash>());


    // Merkle Tree For SHA hashes
    using MERKLE_BYTE_2       = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<2>() + Precision<uint8_t>() + Function<function::Merkle>() + Block());
    using MERKLE_BYTE_4       = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<4>() + Precision<uint8_t>() + Function<function::Merkle>() + Block());
    using MERKLE_BYTE_8       = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<8>() + Precision<uint8_t>() + Function<function::Merkle>() + Block());
    using MERKLE_BYTE_16      = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<16>() + Precision<uint8_t>() + Function<function::Merkle>() + Block());
    using MERKLE_BYTE_32      = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<32>() + Precision<uint8_t>() + Function<function::Merkle>() + Block());
    using MERKLE_BYTE_64      = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<64>() + Precision<uint8_t>() + Function<function::Merkle>() + Block());
    using MERKLE_BYTE_128     = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<128>() + Precision<uint8_t>() + Function<function::Merkle>() + Block());
    using MERKLE_BYTE_256     = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<256>() + Precision<uint8_t>() + Function<function::Merkle>() + Block());
    using MERKLE_BYTE_512     = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<512>() + Precision<uint8_t>() + Function<function::Merkle>() + Block());
    using MERKLE_BYTE_1024    = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<1024>() + Precision<uint8_t>() + Function<function::Merkle>() + Block());
    using MERKLE_BYTE_2048    = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<2048>() + Precision<uint8_t>() + Function<function::Merkle>() + Block());
    using MERKLE_BYTE_4096    = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<4096>() + Precision<uint8_t>() + Function<function::Merkle>() + Block());
    using MERKLE_BYTE_8192    = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<8192>() + Precision<uint8_t>() + Function<function::Merkle>() + Block());
    using MERKLE_BYTE_16384   = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<16384>() + Precision<uint8_t>() + Function<function::Merkle>() + Block());
    using MERKLE_BYTE_32768   = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<32768>() + Precision<uint8_t>() + Function<function::Merkle>() + Block());
    using MERKLE_BYTE_65536   = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<65536>() + Precision<uint8_t>() + Function<function::Merkle>() + Block());
    using MERKLE_BYTE_131072  = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<131072>() + Precision<uint8_t>() + Function<function::Merkle>() + Block());
    using MERKLE_BYTE_262144  = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<262144>() + Precision<uint8_t>() + Function<function::Merkle>() + Block());
    using MERKLE_BYTE_524288  = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<524288>() + Precision<uint8_t>() + Function<function::Merkle>() + Block());
    using MERKLE_BYTE_1048576 = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<1048576>() + Precision<uint8_t>() + Function<function::Merkle>() + Block());
    using MERKLE_BYTE_2097152 = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<2097152>() + Precision<uint8_t>() + Function<function::Merkle>() + Block());
   
    // Merkle Tree For Poseidon2 hashes
    using MERKLE_FIELD_2       = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<2>() + Precision<uint32_t>() + Function<function::Merkle>() + Block());
    using MERKLE_FIELD_4       = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<4>() + Precision<uint32_t>() + Function<function::Merkle>() + Block());
    using MERKLE_FIELD_8       = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<8>() + Precision<uint32_t>() + Function<function::Merkle>() + Block());
    using MERKLE_FIELD_16      = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<16>() + Precision<uint32_t>() + Function<function::Merkle>() + Block());
    using MERKLE_FIELD_32      = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<32>() + Precision<uint32_t>() + Function<function::Merkle>() + Block());
    using MERKLE_FIELD_64      = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<64>() + Precision<uint32_t>() + Function<function::Merkle>() + Block());
    using MERKLE_FIELD_128     = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<128>() + Precision<uint32_t>() + Function<function::Merkle>() + Block());
    using MERKLE_FIELD_256     = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<256>() + Precision<uint32_t>() + Function<function::Merkle>() + Block());
    using MERKLE_FIELD_512     = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<512>() + Precision<uint32_t>() + Function<function::Merkle>() + Block());
    using MERKLE_FIELD_1024    = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<1024>() + Precision<uint32_t>() + Function<function::Merkle>() + Block());
    using MERKLE_FIELD_2048    = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<2048>() + Precision<uint32_t>() + Function<function::Merkle>() + Block());
    using MERKLE_FIELD_4096    = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<4096>() + Precision<uint32_t>() + Function<function::Merkle>() + Block());
    using MERKLE_FIELD_8192    = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<8192>() + Precision<uint32_t>() + Function<function::Merkle>() + Block());
    using MERKLE_FIELD_16384   = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<16384>() + Precision<uint32_t>() + Function<function::Merkle>() + Block());
    using MERKLE_FIELD_32768   = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<32768>() + Precision<uint32_t>() + Function<function::Merkle>() + Block());
    using MERKLE_FIELD_65536   = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<65536>() + Precision<uint32_t>() + Function<function::Merkle>() + Block());
    using MERKLE_FIELD_131072  = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<131072>() + Precision<uint32_t>() + Function<function::Merkle>() + Block());
    using MERKLE_FIELD_262144  = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<262144>() + Precision<uint32_t>() + Function<function::Merkle>() + Block());
    using MERKLE_FIELD_524288  = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<524288>() + Precision<uint32_t>() + Function<function::Merkle>() + Block());
    using MERKLE_FIELD_1048576 = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<1048576>() + Precision<uint32_t>() + Function<function::Merkle>() + Block());
    using MERKLE_FIELD_2097152 = decltype(Algorithm<algorithm::MERKLE>() + MerkleSize<2097152>() + Precision<uint32_t>() + Function<function::Merkle>() + Block());
}

#endif // CUHASH_HPP
