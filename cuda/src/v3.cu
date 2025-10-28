#include <vector>
#include <cupqc.hpp>
#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>
#include <string>
using namespace cupqc;

#define DEBUG_KEY_GEN true // Enable/disable debugging for key generation

using MLDSA65Key = decltype(ML_DSA_65() + Function<function::Keygen>() + Block() + BlockDim<128>()); // Optional operator with default config

using MLDSA65Sign = decltype(ML_DSA_65() + Function<function::Sign>() + Block() + BlockDim<128>()); // Optional operator with default config

using MLDSA65Verify = decltype(ML_DSA_65() + Function<function::Verify>() + Block() + BlockDim<128>()); // Optional operator with default config

__global__ void keygen_kernel_ml_dsa(uint8_t *public_keys, uint8_t *secret_keys, uint8_t *randombytes, uint8_t *workspace) {
    __shared__ uint8_t smem_ptr[MLDSA65Key::shared_memory_size];
    int block = blockIdx.x;
    auto public_key = public_keys + block * MLDSA65Key::public_key_size;
    auto secret_key = secret_keys + block * MLDSA65Key::secret_key_size;
    auto entropy = randombytes + block * MLDSA65Key::entropy_size;
    auto work = workspace + block * MLDSA65Key::workspace_size;

    MLDSA65Key().execute(public_key, secret_key, entropy, work, smem_ptr);
}

__global__ void sign_kernel(uint8_t *signatures, const uint8_t *messages, const size_t message_size, const uint8_t *secret_keys, uint8_t *randombytes, uint8_t *workspace) {
    __shared__ uint8_t smem_ptr[MLDSA65Sign::shared_memory_size];
    int block = blockIdx.x;
    auto signature = signatures + block * ((MLDSA65Sign::signature_size + 7) / 8 * 8);
    auto message = messages + block * message_size;
    auto secret_key = secret_keys + block * MLDSA65Sign::secret_key_size;
    auto entropy = randombytes + block * MLDSA65Sign::entropy_size;
    auto work = workspace + block * MLDSA65Sign::workspace_size;

    MLDSA65Sign().execute(signature, message, message_size, secret_key, entropy, work, smem_ptr);
}

__global__ void verify_kernel(bool *valids, const uint8_t *signatures, const uint8_t *messages, const size_t message_size, const uint8_t *public_keys, uint8_t *workspace) {
    __shared__ uint8_t smem_ptr[MLDSA65Verify::shared_memory_size];
    int block = blockIdx.x;
    auto signature = signatures + block * ((MLDSA65Sign::signature_size + 7) / 8 * 8);
    auto message = messages + block * message_size;
    auto public_key = public_keys + block * MLDSA65Verify::public_key_size;
    auto work = workspace + block * MLDSA65Verify::workspace_size;

    valids[block] = MLDSA65Verify().execute(message, message_size, signature, public_key, work, smem_ptr);
}

double benchmark(const std::string &operation_name, const cudaEvent_t &start, const cudaEvent_t &stop, unsigned int batch) {
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double seconds = milliseconds / 1000.0;
    double throughput = batch / seconds;
    // printf("%s Throughput: %.2f ops/sec\n", operation_name.c_str(), throughput);
    return throughput;
}

double ml_dsa_keygen(std::vector<uint8_t> &public_keys, std::vector<uint8_t> &secret_keys, const unsigned int batch) {
    auto length_public_key = MLDSA65Key::public_key_size;
    auto length_secret_key = MLDSA65Key::secret_key_size;

    auto workspace = make_workspace<MLDSA65Key>(batch);
    auto randombytes = get_entropy<MLDSA65Key>(batch);

    uint8_t *d_public_key = nullptr;
    uint8_t *d_secret_key = nullptr;

    cudaMalloc(reinterpret_cast<void **>(&d_public_key), length_public_key * batch);
    cudaMalloc(reinterpret_cast<void **>(&d_secret_key), length_secret_key * batch);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    keygen_kernel_ml_dsa << <batch, MLDSA65Key::BlockDim >> > (d_public_key, d_secret_key, randombytes, workspace);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(public_keys.data(), d_public_key, length_public_key * batch, cudaMemcpyDeviceToHost);
    cudaMemcpy(secret_keys.data(), d_secret_key, length_secret_key * batch, cudaMemcpyDeviceToHost);

    double throughput = benchmark("Key Generation", start, stop, batch);

    cudaFree(d_public_key);
    cudaFree(d_secret_key);
    destroy_workspace(workspace);
    release_entropy(randombytes);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return throughput;
}

double ml_dsa_sign(std::vector<uint8_t> &signatures, std::vector<uint8_t> &messages, size_t message_size,
    const std::vector<uint8_t> &secret_keys, const unsigned int batch) {
    auto length_secret_key = MLDSA65Sign::secret_key_size;
    auto length_signature = MLDSA65Sign::signature_size;
    // length_signature is not a multiple of 8, need to pad to ensure alignment.
    auto length_signature_padded = ((length_signature + 7) / 8) * 8;

    auto workspace = make_workspace<MLDSA65Sign>(batch);
    auto randombytes = get_entropy<MLDSA65Sign>(batch);
    uint8_t *d_signatures = nullptr;
    uint8_t *d_secret_keys = nullptr;
    uint8_t *d_messages = nullptr;

    cudaMalloc(reinterpret_cast<void **>(&d_signatures), length_signature_padded * batch); // These are uint8_t so length and batch are in bytes
    cudaMalloc(reinterpret_cast<void **>(&d_secret_keys), length_secret_key * batch);
    cudaMalloc(reinterpret_cast<void **>(&d_messages), message_size * batch);

    cudaMemcpy(d_secret_keys, secret_keys.data(), length_secret_key * batch, cudaMemcpyHostToDevice);
    cudaMemcpy(d_messages, messages.data(), message_size * batch, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sign_kernel << <batch, MLDSA65Sign::BlockDim >> > (d_signatures, d_messages, message_size, d_secret_keys, randombytes, workspace);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(signatures.data(), d_signatures, length_signature_padded * batch, cudaMemcpyDeviceToHost);

    double throughput = benchmark("Sign", start, stop, batch);

    cudaFree(d_secret_keys);
    cudaFree(d_signatures);
    cudaFree(d_messages);
    destroy_workspace(workspace);
    release_entropy(randombytes);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return throughput;
}

double ml_dsa_verify(bool *is_valids, const std::vector<uint8_t> &signatures, const std::vector<uint8_t> &messages, size_t message_size,
    const std::vector<uint8_t> &public_keys, const unsigned int batch) {
    auto workspace = make_workspace<MLDSA65Verify>(batch);
    auto length_signature = MLDSA65Verify::signature_size;
    auto length_signature_padded = ((length_signature + 7) / 8) * 8;
    auto length_public_key = MLDSA65Verify::public_key_size;

    uint8_t *d_signatures = nullptr;
    uint8_t *d_messages = nullptr;
    uint8_t *d_public_keys = nullptr;
    bool *d_valids = nullptr;

    cudaMalloc(reinterpret_cast<void **>(&d_signatures), length_signature_padded * batch); // These are uint8_t so length and batch are in bytes
    cudaMalloc(reinterpret_cast<void **>(&d_public_keys), length_public_key * batch);
    cudaMalloc(reinterpret_cast<void **>(&d_messages), message_size * batch);
    cudaMalloc(reinterpret_cast<void **>(&d_valids), batch);

    cudaMemcpy(d_public_keys, public_keys.data(), length_public_key * batch, cudaMemcpyHostToDevice);
    cudaMemcpy(d_signatures, signatures.data(), length_signature_padded * batch, cudaMemcpyHostToDevice);
    cudaMemcpy(d_messages, messages.data(), message_size * batch, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    verify_kernel << <batch, MLDSA65Verify::BlockDim >> > (d_valids, d_signatures, d_messages, message_size, d_public_keys, workspace);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(is_valids, d_valids, batch, cudaMemcpyDeviceToHost);

    double throughput = benchmark("Verification", start, stop, batch);

    cudaFree(d_public_keys);
    cudaFree(d_signatures);
    cudaFree(d_messages);
    cudaFree(d_valids);
    destroy_workspace(workspace);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return throughput;
}

using MLKEM768Key = decltype(ML_KEM_768()
    + Function<function::Keygen>()
    + Block()
    + BlockDim<128>());  // Optional operator with default config

using MLKEM768Encaps = decltype(ML_KEM_768()
    + Function<function::Encaps>()
    + Block()
    + BlockDim<128>());  // Optional operator with default config

using MLKEM768Decaps = decltype(ML_KEM_768()
    + Function<function::Decaps>()
    + Block()
    + BlockDim<128>());  // Optional operator with default config

__global__ void keygen_kernel_ml_kem(uint8_t *public_keys, uint8_t *secret_keys, uint8_t *workspace, uint8_t *randombytes) {
    __shared__ uint8_t smem_ptr[MLKEM768Key::shared_memory_size];
    int block = blockIdx.x;
    auto public_key = public_keys + block * MLKEM768Key::public_key_size;
    auto secret_key = secret_keys + block * MLKEM768Key::secret_key_size;
    auto entropy = randombytes + block * MLKEM768Key::entropy_size;
    auto work = workspace + block * MLKEM768Key::workspace_size;

    MLKEM768Key().execute(public_key, secret_key, entropy, work, smem_ptr);
}

__global__ void encaps_kernel(uint8_t *ciphertexts, uint8_t *shared_secrets, const uint8_t *public_keys, uint8_t *workspace, uint8_t *randombytes) {
    __shared__ uint8_t smem_ptr[MLKEM768Encaps::shared_memory_size];
    int block = blockIdx.x;
    auto shared_secret = shared_secrets + block * MLKEM768Encaps::shared_secret_size;
    auto ciphertext = ciphertexts + block * MLKEM768Encaps::ciphertext_size;
    auto public_key = public_keys + block * MLKEM768Encaps::public_key_size;
    auto entropy = randombytes + block * MLKEM768Encaps::entropy_size;
    auto work = workspace + block * MLKEM768Encaps::workspace_size;

    MLKEM768Encaps().execute(ciphertext, shared_secret, public_key, entropy, work, smem_ptr);
}

__global__ void decaps_kernel(uint8_t *shared_secrets, const uint8_t *ciphertexts, const uint8_t *secret_keys, uint8_t *workspace) {
    __shared__ uint8_t smem_ptr[MLKEM768Decaps::shared_memory_size];
    int block = blockIdx.x;
    auto shared_secret = shared_secrets + block * MLKEM768Decaps::shared_secret_size;
    auto ciphertext = ciphertexts + block * MLKEM768Decaps::ciphertext_size;
    auto secret_key = secret_keys + block * MLKEM768Decaps::secret_key_size;
    auto work = workspace + block * MLKEM768Decaps::workspace_size;

    MLKEM768Decaps().execute(shared_secret, ciphertext, secret_key, work, smem_ptr);
}

double ml_kem_keygen(std::vector<uint8_t> &public_keys, std::vector<uint8_t> &secret_keys, const unsigned int batch) {
    auto length_public_key = MLKEM768Key::public_key_size;
    auto length_secret_key = MLKEM768Key::secret_key_size;

    auto workspace = make_workspace<MLKEM768Key>(batch);
    auto randombytes = get_entropy<MLKEM768Key>(batch);

    uint8_t *d_public_key = nullptr;
    uint8_t *d_secret_key = nullptr;

    cudaMalloc(reinterpret_cast<void **>(&d_public_key), length_public_key * batch);
    cudaMalloc(reinterpret_cast<void **>(&d_secret_key), length_secret_key * batch);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    keygen_kernel_ml_kem << <batch, MLKEM768Key::BlockDim >> > (d_public_key, d_secret_key, workspace, randombytes);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(public_keys.data(), d_public_key, length_public_key * batch, cudaMemcpyDeviceToHost);
    cudaMemcpy(secret_keys.data(), d_secret_key, length_secret_key * batch, cudaMemcpyDeviceToHost);

    double throughput = benchmark("Key Generation", start, stop, batch);

    cudaFree(d_public_key);
    cudaFree(d_secret_key);
    destroy_workspace(workspace);
    release_entropy(randombytes);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return throughput;
}

double ml_kem_encaps(std::vector<uint8_t> &ciphertexts, std::vector<uint8_t> &shared_secrets,
    const std::vector<uint8_t> &public_keys, const unsigned int batch) {
    auto length_ciphertext = MLKEM768Encaps::ciphertext_size;
    auto length_sharedsecret = MLKEM768Encaps::shared_secret_size;
    auto length_public_key = MLKEM768Encaps::public_key_size;

    auto workspace = make_workspace<MLKEM768Encaps>(batch);
    auto randombytes = get_entropy<MLKEM768Encaps>(batch);

    uint8_t *d_ciphertext = nullptr;
    uint8_t *d_sharedsecret = nullptr;
    uint8_t *d_public_key = nullptr;

    cudaMalloc(reinterpret_cast<void **>(&d_ciphertext), length_ciphertext * batch);
    cudaMalloc(reinterpret_cast<void **>(&d_sharedsecret), length_sharedsecret * batch);
    cudaMalloc(reinterpret_cast<void **>(&d_public_key), length_public_key * batch);

    cudaMemcpy(d_public_key, public_keys.data(), length_public_key * batch, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    encaps_kernel << <batch, MLKEM768Encaps::BlockDim >> > (d_ciphertext, d_sharedsecret, d_public_key, workspace, randombytes);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(ciphertexts.data(), d_ciphertext, length_ciphertext * batch, cudaMemcpyDeviceToHost);
    cudaMemcpy(shared_secrets.data(), d_sharedsecret, length_sharedsecret * batch, cudaMemcpyDeviceToHost);

    double throughput = benchmark("Encapsulation", start, stop, batch);

    cudaFree(d_ciphertext);
    cudaFree(d_sharedsecret);
    cudaFree(d_public_key);
    destroy_workspace(workspace);
    release_entropy(randombytes);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return throughput;
}

double ml_kem_decaps(std::vector<uint8_t> &shared_secrets, const std::vector<uint8_t> &ciphertexts,
    const std::vector<uint8_t> &secret_keys, const unsigned int batch) {
    auto length_ciphertext = MLKEM768Decaps::ciphertext_size;
    auto length_sharedsecret = MLKEM768Decaps::shared_secret_size;
    auto length_secret_key = MLKEM768Decaps::secret_key_size;

    auto workspace = make_workspace<MLKEM768Decaps>(batch);

    uint8_t *d_ciphertext = nullptr;
    uint8_t *d_sharedsecret = nullptr;
    uint8_t *d_secret_key = nullptr;

    cudaMalloc(reinterpret_cast<void **>(&d_ciphertext), length_ciphertext * batch);
    cudaMalloc(reinterpret_cast<void **>(&d_sharedsecret), length_sharedsecret * batch);
    cudaMalloc(reinterpret_cast<void **>(&d_secret_key), length_secret_key * batch);

    cudaMemcpy(d_ciphertext, ciphertexts.data(), length_ciphertext * batch, cudaMemcpyHostToDevice);
    cudaMemcpy(d_secret_key, secret_keys.data(), length_secret_key * batch, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    decaps_kernel << <batch, MLKEM768Decaps::BlockDim >> > (d_sharedsecret, d_ciphertext, d_secret_key, workspace);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(shared_secrets.data(), d_sharedsecret, length_sharedsecret * batch, cudaMemcpyDeviceToHost);

    double throughput = benchmark("Decapsulation", start, stop, batch);

    cudaFree(d_ciphertext);
    cudaFree(d_sharedsecret);
    cudaFree(d_secret_key);
    destroy_workspace(workspace);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return throughput;
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[]) {
    unsigned int batch = 50000;
    // ml-dsa
    constexpr size_t message_size = 1024;
    std::vector<uint8_t> public_keys_ml_dsa(MLDSA65Key::public_key_size * batch);
    std::vector<uint8_t> secret_keys_ml_dsa(MLDSA65Key::secret_key_size * batch);
    std::vector<uint8_t> signatures((MLDSA65Sign::signature_size + 7) / 8 * 8 * batch);
    std::vector<uint8_t> messages(message_size * batch);
    bool is_valids[batch];
    //ml-kem
    std::vector<uint8_t> public_keys_ml_kem(MLKEM768Key::public_key_size * batch);
    std::vector<uint8_t> secret_keys_ml_kem(MLKEM768Key::secret_key_size * batch);
    std::vector<uint8_t> ciphertexts(MLKEM768Encaps::ciphertext_size * batch);
    std::vector<uint8_t> encaps_shared_secrets(MLKEM768Encaps::shared_secret_size * batch);
    std::vector<uint8_t> decaps_shared_secrets(MLKEM768Decaps::shared_secret_size * batch);

    while (true) {
        double ml_dsa_keygen_ops = ml_dsa_keygen(public_keys_ml_dsa, secret_keys_ml_dsa, batch);
        double ml_dsa_sign_ops = ml_dsa_sign(signatures, messages, message_size, secret_keys_ml_dsa, batch);
        double ml_dsa_verify_ops = ml_dsa_verify(is_valids, signatures, messages, message_size, public_keys_ml_dsa, batch);

        double ml_kem_keygen_ops = ml_kem_keygen(public_keys_ml_kem, secret_keys_ml_kem, batch);
        double ml_kem_encaps_ops = ml_kem_encaps(ciphertexts, encaps_shared_secrets, public_keys_ml_kem, batch);
        double ml_kem_decaps_ops = ml_kem_decaps(decaps_shared_secrets, ciphertexts, secret_keys_ml_kem, batch);

        printf("DSA_KEYGEN: %.2f DSA_SIGN: %.2f DSA_VERIFY: %.2f KEM_KEYGEN: %.2f KEM_ENCAPS: %.2f KEM_DECAPS: %.2f\n",
            ml_dsa_keygen_ops, ml_dsa_sign_ops, ml_dsa_verify_ops, ml_kem_keygen_ops, ml_kem_encaps_ops, ml_kem_decaps_ops);
    }
}
