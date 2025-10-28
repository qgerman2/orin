#ifndef CUHASH_DATABASE_MERKLE_CUH
#define CUHASH_DATABASE_MERKLE_CUH

#include <cstdint>
#include "operators.hpp"
#include "cuhash_types.hpp"

namespace cupqc {
    namespace database {
        template<class Hash, typename Precision>
        __device__ void create_leaf(Precision* leaf, const Precision* data, Hash& hash, const size_t inbuf_len);

        template<uint32_t N, class Hash, typename Precision>
        __device__ void generate_tree(const Precision* leaves, Hash& hash, tree<N, Hash, Precision>& merkle);

        template<uint32_t N, class Hash, typename Precision>
         __device__ void generate_proof(proof<N, Hash, Precision>& proof, const Precision* proof_leaf, const uint32_t leaf_index, const tree<N, Hash, Precision>& merkle);

        template<uint32_t N, class Hash, typename Precision>
        __device__ bool verify_proof(const proof<N, Hash, Precision>& proof, const Precision* verify_leaf, const uint32_t leaf_index, const Precision* root, Hash& hash);
    }
}

#endif // CUHASH_DATABASE_MERKLE_CUH