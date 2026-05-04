#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "seal/seal.h"

using namespace std;
using namespace seal;
using clk = std::chrono::steady_clock;

static string sec_level_to_string(sec_level_type level) {
    switch (level) {
    case sec_level_type::none:  return "none";
    case sec_level_type::tc128: return "tc128";
    case sec_level_type::tc192: return "tc192";
    case sec_level_type::tc256: return "tc256";
    default: return "unknown";
    }
}

// =============================================================================
// Schemes we explicitly skip in the comparison.
// =============================================================================

static const std::vector<std::string> SKIP_SCHEMES = {
    "ls_poly_2", "ls_poly_3", "ls_poly_5", "ls_poly_7"
};

static bool scheme_is_skipped(const std::string& name) {
    return std::find(SKIP_SCHEMES.begin(), SKIP_SCHEMES.end(), name)
        != SKIP_SCHEMES.end();
}

// =============================================================================
// Hash for SEAL parms_id_type (used to key the W2 plaintext cache by level).
// =============================================================================

struct ParmsIdHasher {
    size_t operator()(const parms_id_type& pid) const noexcept {
        size_t h = 0;
        for (auto v : pid) {
            h ^= std::hash<uint64_t>{}(v)+0x9e3779b97f4a7c15ULL
                + (h << 6) + (h >> 2);
        }
        return h;
    }
};

// =============================================================================
// Data structures (model + scheme definitions)
// =============================================================================

struct ModelMeta {
    size_t input_dim = 0;       // *** holds the padded value at runtime ***
    size_t hidden_dim = 0;
    size_t n_test = 0;
    size_t num_classes = 0;
    size_t n_schemes = 0;
};

struct SchemeConfig {
    string name;
    int    degree = 0;
};

struct SchemeResults {
    string name;
    int    degree = 0;
    int    activation_depth = 0;
    int    total_circuit_depth = 0;
    vector<double> coeffs;
    vector<double> plain_logits;       // n_test * num_classes, row-major

    size_t n = 0;
    size_t plain_correct_vs_true = 0;
    size_t he_correct_vs_true = 0;
    size_t plain_correct_vs_relu = 0;
    size_t he_correct_vs_relu = 0;
    size_t plain_he_agree = 0;

    double sum_abs_err = 0.0;
    double max_abs_err = 0.0;
    double max_plain_cpp_err = 0.0;
    size_t n_logits = 0;

    double t_enc = 0.0, t_matvec = 0.0, t_powers = 0.0;
    double t_act = 0.0, t_out = 0.0, t_dec = 0.0;
};

struct SchemeAcc {
    size_t n = 0;
    size_t plain_correct_vs_true = 0;
    size_t he_correct_vs_true = 0;
    size_t plain_correct_vs_relu = 0;
    size_t he_correct_vs_relu = 0;
    size_t plain_he_agree = 0;
    double sum_abs_err = 0.0;
    double max_abs_err = 0.0;
    double max_plain_cpp_err = 0.0;
    size_t n_logits = 0;
    double t_enc = 0.0, t_matvec = 0.0, t_powers = 0.0;
    double t_act = 0.0, t_out = 0.0, t_dec = 0.0;
};

struct DebugSample {
    size_t idx = 0;
    string text;
};

// =============================================================================
// Loaders
// =============================================================================

static ModelMeta load_meta(const string& path) {
    ifstream in(path);
    if (!in) throw runtime_error("Cannot open meta file: " + path);
    ModelMeta m;
    string key;
    while (in >> key) {
        if (key == "input_dim")   in >> m.input_dim;
        else if (key == "hidden_dim")  in >> m.hidden_dim;
        else if (key == "n_test")      in >> m.n_test;
        else if (key == "num_classes") in >> m.num_classes;
        else if (key == "n_schemes")   in >> m.n_schemes;
        else { string tmp; in >> tmp; }
    }
    if (m.num_classes == 0)
        throw runtime_error("meta.txt is missing 'num_classes'");
    return m;
}

static vector<SchemeConfig> load_schemes(const string& path) {
    ifstream in(path);
    if (!in) throw runtime_error("Cannot open schemes file: " + path);
    vector<SchemeConfig> v;
    string line;
    while (getline(in, line)) {
        size_t p = line.find_first_not_of(" \t\r\n");
        if (p == string::npos) continue;
        if (line[p] == '#') continue;
        istringstream iss(line);
        SchemeConfig s;
        if (!(iss >> s.name >> s.degree))
            throw runtime_error("Malformed schemes.txt line: " + line);
        v.push_back(s);
    }
    return v;
}

template<typename T>
static vector<T> load_bin(const string& path, size_t count) {
    ifstream in(path, ios::binary);
    if (!in) throw runtime_error("Cannot open binary file: " + path);
    vector<T> buf(count);
    in.read(reinterpret_cast<char*>(buf.data()),
        static_cast<streamsize>(count * sizeof(T)));
    if (!in) throw runtime_error("Short read on " + path);
    return buf;
}

// =============================================================================
// Encoding helpers — batched / BSGS layout
// =============================================================================

static bool is_power_of_two(size_t x) { return x && ((x & (x - 1)) == 0); }

// case_a builder (kept for clarity / backward compat). Equivalent to
// build_diagonal_batched_chunk(W, m, n, /*chunk_offset=*/0, /*chunk_dim=*/n,
//                              k, B, slot_count).
static vector<double> build_diagonal_batched(const vector<double>& W,
    size_t m, size_t n, size_t k,
    size_t B, size_t slot_count)
{
    vector<double> diag(slot_count, 0.0);
    for (size_t i = 0; i < m; ++i) {
        const double w = W[i * n + (i + k) % n];
        const size_t base = i * B;
        for (size_t b = 0; b < B; ++b) diag[base + b] = w;
    }
    return diag;
}

// Chunked diagonal builder for case_b (input_dim > hidden_dim).
//   - hidden_dim:     m, the BSGS ciphertext "row" dim (= slot_count / B)
//   - input_dim_total: full W1 column count (i.e. meta.input_dim)
//   - chunk_offset:   starting column in W1 for this chunk
//   - chunk_dim:      number of columns this chunk handles (= hidden_dim)
//   - k:              BSGS diagonal index within this chunk, k in [0, chunk_dim)
//
// Diagonal d_k[i] = W[i, chunk_offset + ((i + k) % chunk_dim)].
static vector<double> build_diagonal_batched_chunk(
    const vector<double>& W,
    size_t hidden_dim, size_t input_dim_total,
    size_t chunk_offset, size_t chunk_dim,
    size_t k, size_t B, size_t slot_count)
{
    vector<double> diag(slot_count, 0.0);
    for (size_t i = 0; i < hidden_dim; ++i) {
        size_t col_local = (i + k) % chunk_dim;
        size_t col = chunk_offset + col_local;
        // Defensive: W1 was zero-padded out to input_dim_total, so OOB cols
        // (shouldn't occur if (chunk_offset+chunk_dim) <= input_dim_total)
        // simply contribute zero.
        if (col >= input_dim_total) continue;
        const double w = W[i * input_dim_total + col];
        const size_t base = i * B;
        if (base + B > slot_count) break;
        for (size_t b = 0; b < B; ++b) diag[base + b] = w;
    }
    return diag;
}

static vector<double> plaintext_rotate(const vector<double>& v,
    long long shift, size_t slot_count) {
    vector<double> out(slot_count, 0.0);
    long long s = shift % static_cast<long long>(slot_count);
    if (s < 0) s += slot_count;
    for (size_t i = 0; i < slot_count; ++i) {
        out[i] = v[(i + static_cast<size_t>(s)) % slot_count];
    }
    return out;
}

// case_a packing (kept for backward compat). Used when n_chunks == 1.
static vector<double> pack_batch_interleaved(const vector<vector<double>>& xs,
    size_t input_dim, size_t B, size_t slot_count)
{
    vector<double> packed(slot_count, 0.0);
    const size_t real = std::min(xs.size(), B);
    for (size_t s = 0; s < slot_count; ++s) {
        size_t b = s % B;
        size_t feat = (s / B) % input_dim;
        if (b < real) packed[s] = xs[b][feat];
    }
    return packed;
}

// Chunked packing for case_b: pack only feats [feat_offset, feat_offset+chunk_dim)
// into the slot space, with slot index s = i*B + b corresponding to
// xs[b][feat_offset + i]  for i in [0, chunk_dim) and B-fold replication
// across the slot space (since slot_count = chunk_dim * B in case_b).
static vector<double> pack_batch_chunk_interleaved(
    const vector<vector<double>>& xs,
    size_t feat_offset, size_t chunk_dim,
    size_t B, size_t slot_count)
{
    vector<double> packed(slot_count, 0.0);
    const size_t real = std::min(xs.size(), B);
    for (size_t s = 0; s < slot_count; ++s) {
        size_t b = s % B;
        size_t feat_local = (s / B) % chunk_dim;
        if (b < real) packed[s] = xs[b][feat_offset + feat_local];
    }
    return packed;
}

static vector<double> pad_per_row_batched(const vector<double>& v,
    size_t B, size_t slot_count)
{
    vector<double> out(slot_count, 0.0);
    for (size_t i = 0; i < v.size(); ++i) {
        const size_t base = i * B;
        if (base + B > slot_count) break;
        for (size_t b = 0; b < B; ++b) out[base + b] = v[i];
    }
    return out;
}


// =============================================================================
// Theoretical operation-count diagnostics per encrypted batch
// =============================================================================

struct CKKSOpCount {
    size_t encryptions = 0;
    size_t decryptions = 0;
    size_t ct_ct_mult = 0;
    size_t ct_pt_mult = 0;
    size_t rotations = 0;
    size_t rescale = 0;
    size_t relinearize = 0;
    size_t pt_add = 0;
    size_t ct_add = 0;
    size_t preencoded_w1_plaintexts = 0;
    size_t preencoded_w2_plaintexts = 0;
};

static size_t count_baby_rotations(size_t n1, size_t B, size_t slot_count) {
    size_t out = 0;
    for (size_t j = 1; j < n1; ++j)
        if ((j * B) % slot_count != 0) ++out;
    return out;
}

static size_t count_giant_rotations(size_t chunk_dim, size_t n1, size_t n2,
                                    size_t B, size_t slot_count) {
    size_t out = 0;
    for (size_t i = 1; i < n2; ++i) {
        if (i * n1 >= chunk_dim) break;
        if ((i * n1 * B) % slot_count != 0) ++out;
    }
    return out;
}

static size_t count_output_tree_rotations(size_t hidden_dim, size_t B, size_t slot_count) {
    size_t out = 0;
    for (size_t s = hidden_dim / 2; s >= 1; s /= 2) {
        if ((s * B) % slot_count != 0) ++out;
        if (s == 1) break;
    }
    return out;
}

static CKKSOpCount activation_op_count(const vector<double>& coeffs) {
    CKKSOpCount oc;
    const int deg = static_cast<int>(coeffs.size()) - 1;
    auto nonzero = [](double x) { return std::fabs(x) > 0.0; };
    if (deg == 2) {
        if (coeffs.size() > 2 && nonzero(coeffs[2])) { oc.ct_pt_mult++; oc.rescale++; }
        if (coeffs.size() > 1 && nonzero(coeffs[1])) { oc.ct_pt_mult++; oc.rescale++; }
        if (coeffs.size() > 0 && nonzero(coeffs[0])) { oc.pt_add++; }
    } else if (deg == 3) {
        oc.ct_pt_mult += 2; oc.rescale += 2; // two affine_in_y blocks
        if (nonzero(coeffs[2])) oc.pt_add++;
        if (nonzero(coeffs[0])) oc.pt_add++;
        oc.ct_ct_mult += 1; oc.relinearize += 1; oc.rescale += 1;
        oc.ct_add += 1;
    } else if (deg == 5) {
        oc.ct_pt_mult += 3; oc.rescale += 3; // three affine_in_y blocks
        if (nonzero(coeffs[2])) oc.pt_add++;
        if (nonzero(coeffs[0])) oc.pt_add++;
        if (nonzero(coeffs[4])) oc.pt_add++;
        oc.ct_ct_mult += 2; oc.relinearize += 2; oc.rescale += 2;
        oc.ct_add += 2;
    } else if (deg == 7) {
        oc.ct_pt_mult += 4; oc.rescale += 4; // four affine_in_y blocks
        if (nonzero(coeffs[2])) oc.pt_add++;
        if (nonzero(coeffs[0])) oc.pt_add++;
        if (nonzero(coeffs[6])) oc.pt_add++;
        if (nonzero(coeffs[4])) oc.pt_add++;
        oc.ct_ct_mult += 3; oc.relinearize += 3; oc.rescale += 3;
        oc.ct_add += 3;
    }
    return oc;
}

static CKKSOpCount estimate_ops_per_batch(
    size_t input_dim, size_t hidden_dim, size_t num_classes,
    size_t B, size_t slot_count, size_t n1, size_t n2,
    int max_degree_for_common_powers, const vector<double>& coeffs)
{
    CKKSOpCount oc;
    const bool case_b = (input_dim > hidden_dim) && (input_dim % hidden_dim == 0);
    const size_t n_chunks = case_b ? (input_dim / hidden_dim) : 1;
    const size_t chunk_dim = case_b ? hidden_dim : input_dim;

    oc.encryptions = n_chunks;
    oc.decryptions = num_classes;

    // First affine layer, BSGS diagonal method. One rescale after all chunk sums.
    oc.ct_pt_mult += n_chunks * chunk_dim;
    oc.rotations += n_chunks * (
        count_baby_rotations(n1, B, slot_count) +
        count_giant_rotations(chunk_dim, n1, n2, B, slot_count));
    oc.rescale += 1;
    oc.pt_add += 1; // b1
    oc.preencoded_w1_plaintexts = n_chunks * n1 * n2;

    // Common powers block used in the comparison loop.
    oc.ct_ct_mult += 1; oc.relinearize += 1; oc.rescale += 1; // y^2
    if (max_degree_for_common_powers >= 5) {
        oc.ct_ct_mult += 1; oc.relinearize += 1; oc.rescale += 1; // y^4
    }

    // Scheme-specific polynomial activation.
    CKKSOpCount act = activation_op_count(coeffs);
    oc.ct_ct_mult += act.ct_ct_mult;
    oc.ct_pt_mult += act.ct_pt_mult;
    oc.rotations += act.rotations;
    oc.rescale += act.rescale;
    oc.relinearize += act.relinearize;
    oc.pt_add += act.pt_add;
    oc.ct_add += act.ct_add;

    // Output layer: one ct-pt multiply and hidden-coordinate reduction per class.
    const size_t out_rot = count_output_tree_rotations(hidden_dim, B, slot_count);
    oc.ct_pt_mult += num_classes;
    oc.rotations += num_classes * out_rot;
    oc.rescale += num_classes;
    oc.pt_add += num_classes; // b2
    oc.preencoded_w2_plaintexts = num_classes;
    return oc;
}

static void print_operation_counts(
    const vector<SchemeResults>& schemes, const ModelMeta& meta,
    size_t B, size_t slot_count, size_t n1, size_t n2, int max_degree)
{
    cout << "\n" << string(78, '=') << "\n"
         << "Theoretical CKKS operation counts per encrypted batch" << "\n"
         << string(78, '=') << "\n";
    cout << "Counts follow the implemented schedule: input BSGS matvec, common powers "
         << "block, scheme-specific polynomial activation, and per-class output reduction.\n";
    cout << "  " << left << setw(14) << "Scheme" << right
         << setw(8) << "Enc"
         << setw(8) << "Dec"
         << setw(10) << "ct-ct"
         << setw(10) << "ct-pt"
         << setw(10) << "Rot"
         << setw(10) << "Rescale"
         << setw(10) << "Relin"
         << setw(10) << "PtAdd" << "\n";
    cout << "  " << string(86, '-') << "\n";
    for (const auto& s : schemes) {
        CKKSOpCount oc = estimate_ops_per_batch(
            meta.input_dim, meta.hidden_dim, meta.num_classes,
            B, slot_count, n1, n2, max_degree, s.coeffs);
        cout << "  " << left << setw(14) << s.name << right
             << setw(8) << oc.encryptions
             << setw(8) << oc.decryptions
             << setw(10) << oc.ct_ct_mult
             << setw(10) << oc.ct_pt_mult
             << setw(10) << oc.rotations
             << setw(10) << oc.rescale
             << setw(10) << oc.relinearize
             << setw(10) << oc.pt_add << "\n";
    }
    const bool case_b = (meta.input_dim > meta.hidden_dim) &&
                        (meta.input_dim % meta.hidden_dim == 0);
    const size_t n_chunks = case_b ? (meta.input_dim / meta.hidden_dim) : 1;
    cout << "  Pre-encoded W1 diagonals per level = " << (n_chunks * n1 * n2)
         << "; W2 plaintexts per post-activation level = " << meta.num_classes
         << "; n_chunks=" << n_chunks << ".\n";
}

// =============================================================================
// Compile-time depth accounting for our polynomial evaluator
// =============================================================================

static int poly_activation_depth(int deg) {
    if (deg == 2 || deg == 3) return 2;
    if (deg == 5 || deg == 7) return 3;
    throw runtime_error("Unsupported polynomial degree: " + to_string(deg));
}

// =============================================================================
// CKKS runtime  (now parameterized by N, scale, coeff_bits)
// =============================================================================

class HECKKSRuntime {
public:
    static constexpr double DEFAULT_SCALE = 1099511627776.0;  // 2^40

    size_t                   poly_degree;
    shared_ptr<SEALContext>  context;
    size_t                   slot_count;
    double                   scale;

    SecretKey                secret_key;
    PublicKey                public_key;
    shared_ptr<RelinKeys>    relin_keys;
    shared_ptr<GaloisKeys>   galois_keys;

    shared_ptr<CKKSEncoder>  encoder;
    shared_ptr<Encryptor>    encryptor;
    shared_ptr<Decryptor>    decryptor;
    shared_ptr<Evaluator>    evaluator;

    size_t max_rescales_available = 0;

    HECKKSRuntime(const std::vector<int>& rotation_steps,
                  int scale_log2 = 40,
                  std::vector<int> coeff_bits = {60, 40, 40, 40, 40, 40, 60},
                  size_t poly_degree_arg = 16384)
        : poly_degree(poly_degree_arg)
    {
        EncryptionParameters parms(scheme_type::ckks);
        parms.set_poly_modulus_degree(poly_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_degree, coeff_bits));

        context = make_shared<SEALContext>(parms, true, sec_level_type::tc128);
        slot_count = poly_degree / 2;
        scale = std::pow(2.0, static_cast<double>(scale_log2));

        auto key_data = context->key_context_data();
        if (!key_data || !key_data->qualifiers().parameters_set()) {
            throw runtime_error("Invalid CKKS parameters under SEAL tc128 security check");
        }
        const auto actual_sec_level = key_data->qualifiers().sec_level;
        if (actual_sec_level != sec_level_type::tc128) {
            throw runtime_error("CKKS parameters do not satisfy requested tc128 security; actual="
                + sec_level_to_string(actual_sec_level));
        }

        KeyGenerator keygen(*context);
        secret_key = keygen.secret_key();
        keygen.create_public_key(public_key);
        RelinKeys rk;   keygen.create_relin_keys(rk);

        GaloisKeys gk;
        keygen.create_galois_keys(rotation_steps, gk);

        relin_keys = make_shared<RelinKeys>(std::move(rk));
        galois_keys = make_shared<GaloisKeys>(std::move(gk));

        encoder = make_shared<CKKSEncoder>(*context);
        encryptor = make_shared<Encryptor>(*context, public_key);
        decryptor = make_shared<Decryptor>(*context, secret_key);
        evaluator = make_shared<Evaluator>(*context, *encoder);

        auto ctx = context->first_context_data();
        size_t data_primes = 0;
        while (ctx) { ++data_primes; ctx = ctx->next_context_data(); }
        max_rescales_available = data_primes > 0 ? data_primes - 1 : 0;

        cout << "[SEAL] CKKS parameters:\n"
            << "   poly_modulus_degree   = " << poly_degree << "\n"
            << "   slot_count            = " << slot_count << "\n"
            << "   coeff_modulus bits    = {";
        for (size_t i = 0; i < coeff_bits.size(); ++i)
            cout << (i ? ", " : "") << coeff_bits[i];
        const int total_log_q = std::accumulate(coeff_bits.begin(), coeff_bits.end(), 0);
        const size_t max_logq_tc128 = CoeffModulus::MaxBitCount(poly_degree, sec_level_type::tc128);
        cout << "}\n"
            << "   total log_q           = " << total_log_q << "\n"
            << "   requested security    = tc128\n"
            << "   SEAL security level   = " << sec_level_to_string(actual_sec_level) << "\n"
            << "   MaxBitCount(tc128)    = " << max_logq_tc128 << " bits\n"
            << "   scale                 = 2^" << scale_log2 << "\n"
            << "   rescales available    = " << max_rescales_available << "\n"
            << "   galois rotation steps = " << rotation_steps.size() << "\n";
    }

    Plaintext encode_scalar_at(double val, const parms_id_type& pid) const {
        Plaintext pt;
        encoder->encode(val, pid, scale, pt);
        return pt;
    }

    Plaintext encode_vec_at(const vector<double>& v,
        const parms_id_type& pid) const {
        Plaintext pt;
        encoder->encode(v, pid, scale, pt);
        return pt;
    }

    std::vector<parms_id_type> build_level_chain() const {
        std::vector<parms_id_type> chain;
        auto cd = context->first_context_data();
        while (cd) {
            chain.push_back(cd->parms_id());
            cd = cd->next_context_data();
        }
        return chain;
    }
};

// =============================================================================
// Polynomial activation via Paterson-Stockmeyer
// =============================================================================

class PolyEvaluator {
public:
    PolyEvaluator(HECKKSRuntime& rt) : rt_(rt) {}

    Ciphertext eval(const Ciphertext& y,
        const Ciphertext& y2,
        const Ciphertext& y4,
        const vector<double>& c) const {
        int deg = static_cast<int>(c.size()) - 1;
        switch (deg) {
        case 2: return eval_deg2(y, y2, c);
        case 3: return eval_deg3(y, y2, c);
        case 5: return eval_deg5(y, y2, y4, c);
        case 7: return eval_deg7(y, y2, y4, c);
        default:
            throw runtime_error("PolyEvaluator: unsupported degree "
                + to_string(deg));
        }
    }

private:
    HECKKSRuntime& rt_;

    Ciphertext affine_in_y(const Ciphertext& y, double c_hi, double c_lo) const {
        auto& ev = *rt_.evaluator;
        if (c_hi == 0.0) {
            throw runtime_error("affine_in_y: c_hi=0 would produce a "
                "transparent ciphertext.");
        }
        Plaintext pt_hi = rt_.encode_scalar_at(c_hi, y.parms_id());
        Ciphertext A;
        ev.multiply_plain(y, pt_hi, A);
        ev.rescale_to_next_inplace(A);
        A.scale() = rt_.scale;
        if (c_lo != 0.0) {
            Plaintext pt_lo = rt_.encode_scalar_at(c_lo, A.parms_id());
            ev.add_plain_inplace(A, pt_lo);
        }
        return A;
    }

    void add_aligned(Ciphertext& t, Ciphertext src) const {
        auto& ev = *rt_.evaluator;
        if (src.parms_id() != t.parms_id())
            ev.mod_switch_to_inplace(src, t.parms_id());
        src.scale() = t.scale();
        ev.add_inplace(t, src);
    }

    Ciphertext mod_switched_copy(const Ciphertext& ct,
        parms_id_type target) const {
        Ciphertext out = ct;
        if (out.parms_id() != target)
            rt_.evaluator->mod_switch_to_inplace(out, target);
        out.scale() = rt_.scale;
        return out;
    }

    Ciphertext eval_deg2(const Ciphertext& y,
        const Ciphertext& y2,
        const vector<double>& c) const {
        auto& ev = *rt_.evaluator;

        Ciphertext T;
        bool T_init = false;

        if (c[2] != 0.0) {
            Plaintext pt = rt_.encode_scalar_at(c[2], y2.parms_id());
            ev.multiply_plain(y2, pt, T);
            ev.rescale_to_next_inplace(T);
            T.scale() = rt_.scale;
            T_init = true;
        }

        if (c[1] != 0.0) {
            Plaintext pt = rt_.encode_scalar_at(c[1], y.parms_id());
            Ciphertext S;
            ev.multiply_plain(y, pt, S);
            ev.rescale_to_next_inplace(S);
            S.scale() = rt_.scale;
            if (T_init) add_aligned(T, std::move(S));
            else { T = std::move(S); T_init = true; }
        }

        if (!T_init) {
            throw runtime_error("eval_deg2: c[1]=c[2]=0 (constant poly is "
                "not supported)");
        }

        if (c[0] != 0.0) {
            Plaintext pt_c0 = rt_.encode_scalar_at(c[0], T.parms_id());
            ev.add_plain_inplace(T, pt_c0);
        }
        return T;
    }

    Ciphertext eval_deg3(const Ciphertext& y,
        const Ciphertext& y2,
        const vector<double>& c) const {
        auto& ev = *rt_.evaluator;
        Ciphertext A = affine_in_y(y, c[3], c[2]);
        Ciphertext B = affine_in_y(y, c[1], c[0]);
        Ciphertext A_ms = mod_switched_copy(A, y2.parms_id());
        Ciphertext p;
        ev.multiply(A_ms, y2, p);
        ev.relinearize_inplace(p, *rt_.relin_keys);
        ev.rescale_to_next_inplace(p);
        p.scale() = rt_.scale;
        add_aligned(p, std::move(B));
        return p;
    }

    Ciphertext eval_deg5(const Ciphertext& y,
        const Ciphertext& y2,
        const Ciphertext& y4,
        const vector<double>& c) const {
        auto& ev = *rt_.evaluator;
        Ciphertext A = affine_in_y(y, c[3], c[2]);
        Ciphertext B = affine_in_y(y, c[1], c[0]);
        Ciphertext A_ms = mod_switched_copy(A, y2.parms_id());
        Ciphertext q_low;
        ev.multiply(A_ms, y2, q_low);
        ev.relinearize_inplace(q_low, *rt_.relin_keys);
        ev.rescale_to_next_inplace(q_low);
        q_low.scale() = rt_.scale;
        add_aligned(q_low, std::move(B));

        Ciphertext q_high = affine_in_y(y, c[5], c[4]);
        Ciphertext q_high_ms = mod_switched_copy(q_high, y4.parms_id());
        Ciphertext p;
        ev.multiply(q_high_ms, y4, p);
        ev.relinearize_inplace(p, *rt_.relin_keys);
        ev.rescale_to_next_inplace(p);
        p.scale() = rt_.scale;
        add_aligned(p, std::move(q_low));
        return p;
    }

    Ciphertext eval_deg7(const Ciphertext& y,
        const Ciphertext& y2,
        const Ciphertext& y4,
        const vector<double>& c) const {
        auto& ev = *rt_.evaluator;
        auto build_cubic_block = [&](double a3, double a2, double a1, double a0) {
            Ciphertext A = affine_in_y(y, a3, a2);
            Ciphertext B = affine_in_y(y, a1, a0);
            Ciphertext A_ms = mod_switched_copy(A, y2.parms_id());
            Ciphertext q;
            ev.multiply(A_ms, y2, q);
            ev.relinearize_inplace(q, *rt_.relin_keys);
            ev.rescale_to_next_inplace(q);
            q.scale() = rt_.scale;
            add_aligned(q, std::move(B));
            return q;
            };
        Ciphertext q_low = build_cubic_block(c[3], c[2], c[1], c[0]);
        Ciphertext q_high = build_cubic_block(c[7], c[6], c[5], c[4]);
        q_high.scale() = rt_.scale;
        Ciphertext p;
        ev.multiply(q_high, y4, p);
        ev.relinearize_inplace(p, *rt_.relin_keys);
        ev.rescale_to_next_inplace(p);
        p.scale() = rt_.scale;
        add_aligned(p, std::move(q_low));
        return p;
    }
};

// =============================================================================
// Inference engine — batched + BSGS, multiclass output
// =============================================================================
//
// Two regimes:
//   case_a:  input_dim <= hidden_dim and hidden_dim % input_dim == 0
//            (e.g. Otto: 128 / 256). Single chunk; legacy code path.
//   case_b:  input_dim >  hidden_dim and input_dim  % hidden_dim == 0
//            (e.g. FGVC DINOv2: 768 / 256). Input is split into
//            n_chunks_ = input_dim / hidden_dim ciphertexts, each
//            packing chunk_dim_ = hidden_dim consecutive features.
//            BSGS is run independently per chunk against a sliced W1
//            and the chunk results are accumulated. Each chunk is
//            internally a "case_a" problem with input_dim == hidden_dim,
//            so it uses the same diagonal/giant-step recipe.
//
// In both cases:
//   chunk_dim_ = hidden_dim for case_b, input_dim for case_a
//   bsgs_dim_  = chunk_dim_  (the BSGS factorization target)
//   n_chunks_  = 1 for case_a, input_dim/hidden_dim for case_b
// =============================================================================

class InferenceEngine {
public:
    // Compute the full set of rotation steps the engine will need. Pass
    // bsgs_dim explicitly because for case_b the BSGS is over hidden_dim,
    // not input_dim. The output-tree rotations only depend on hidden_dim and B.
    static std::vector<int> compute_rotation_steps(
        size_t B, size_t hidden_dim, size_t n1, size_t n2,
        size_t slot_count)
    {
        auto reduce = [slot_count](size_t s) -> int {
            size_t r = s % slot_count;
            return static_cast<int>(r);  // 0 means identity, will be filtered
        };

        std::vector<int> steps;
        for (size_t j = 1; j < n1; ++j) {
            int r = reduce(j * B);
            if (r != 0) steps.push_back(r);
        }
        for (size_t i = 1; i < n2; ++i) {
            int r = reduce(i * n1 * B);
            if (r != 0) steps.push_back(r);
        }
        for (size_t s = 1; s <= hidden_dim / 2; s *= 2) {
            int r = reduce(s * B);
            if (r != 0) steps.push_back(r);
        }
        std::sort(steps.begin(), steps.end());
        steps.erase(std::unique(steps.begin(), steps.end()), steps.end());
        return steps;
    }

    InferenceEngine(HECKKSRuntime& rt,
        const ModelMeta& meta,
        vector<double> W1,
        vector<double> b1,
        vector<double> W2,
        vector<double> b2,
        size_t n1, size_t n2,
        const std::set<int>& post_act_levels)
        : rt_(rt), meta_(meta),
        W1_(std::move(W1)), b1_(std::move(b1)),
        W2_(std::move(W2)), b2_(std::move(b2)),
        poly_(rt_),
        n1_(n1), n2_(n2)
    {
        if (!is_power_of_two(meta_.hidden_dim))
            throw runtime_error("hidden_dim must be a power of two");
        if (rt_.slot_count % meta_.hidden_dim != 0)
            throw runtime_error("slot_count must be a multiple of hidden_dim");
        B_ = rt_.slot_count / meta_.hidden_dim;
        if (!is_power_of_two(B_))
            throw runtime_error("Derived batch size B must be a power of two");

        // BSGS cyclic correctness: input_dim and hidden_dim must be
        // commensurable. We accept either:
        //   (a) input_dim <= hidden_dim and hidden_dim % input_dim == 0
        //   (b) input_dim >  hidden_dim and input_dim  % hidden_dim == 0
        // The padding logic in main() ensures one of these holds.
        const bool case_a = (meta_.input_dim <= meta_.hidden_dim) &&
                            (meta_.hidden_dim % meta_.input_dim == 0);
        const bool case_b = (meta_.input_dim >  meta_.hidden_dim) &&
                            (meta_.input_dim % meta_.hidden_dim == 0);
        if (!case_a && !case_b)
            throw runtime_error(
                "input_dim and hidden_dim must be commensurable for BSGS "
                "cyclic correctness (input_dim=" + to_string(meta_.input_dim) +
                ", hidden_dim=" + to_string(meta_.hidden_dim) + ")");

        is_case_b_ = case_b;
        chunk_dim_ = is_case_b_ ? meta_.hidden_dim : meta_.input_dim;
        n_chunks_  = is_case_b_ ? (meta_.input_dim / meta_.hidden_dim) : 1;

        if (n1_ * n2_ < chunk_dim_)
            throw runtime_error("BSGS factorization n1*n2 must be >= chunk_dim "
                "(chunk_dim=" + to_string(chunk_dim_) + ", n1*n2="
                + to_string(n1_ * n2_) + ")");

        if (W1_.size() != meta_.hidden_dim * meta_.input_dim)
            throw runtime_error("W1 size mismatch");
        if (W2_.size() != meta_.num_classes * meta_.hidden_dim)
            throw runtime_error("W2 size mismatch (expect num_classes*hidden_dim)");
        if (b2_.size() != meta_.num_classes)
            throw runtime_error("b2 size mismatch (expect num_classes)");

        cout << "[Engine] B=" << B_ << "  n1=" << n1_ << "  n2=" << n2_
            << "  C=" << meta_.num_classes
            << "  n_chunks=" << n_chunks_
            << "  chunk_dim=" << chunk_dim_
            << "  (case_" << (is_case_b_ ? "b" : "a") << ")\n"
            << "[Engine] matvec rotations per chunk: " << (n1_ + n2_ - 2)
            << ", per-class output tree rotations: "
            << static_cast<int>(std::log2(meta_.hidden_dim)) << "\n";

        precompute_layer_plaintexts(post_act_levels);
    }

    size_t batch_size() const { return B_; }
    size_t n_chunks()   const { return n_chunks_; }

    // Encrypt one batch. Returns one ciphertext per input chunk.
    vector<Ciphertext> encrypt_input_batched(
        const vector<vector<double>>& xs, double& batch_t_enc) const
    {
        auto t0 = clk::now();
        vector<Ciphertext> cts(n_chunks_);
        for (size_t g = 0; g < n_chunks_; ++g) {
            const size_t feat_offset = is_case_b_ ? g * chunk_dim_ : 0;
            vector<double> packed = pack_batch_chunk_interleaved(
                xs, feat_offset, chunk_dim_, B_, rt_.slot_count);
            Plaintext pt;
            rt_.encoder->encode(packed, rt_.scale, pt);
            rt_.encryptor->encrypt(pt, cts[g]);
        }
        batch_t_enc = chrono::duration<double, milli>(clk::now() - t0).count();
        return cts;
    }

    // Layer-1 matvec: W1 * x + b1 over all chunks, summed.
    Ciphertext layer1_matvec(const vector<Ciphertext>& ct_xs,
                             double& matvec_ms) const {
        auto& ev = *rt_.evaluator;
        auto t0 = clk::now();

        Ciphertext acc;
        bool acc_set = false;

        for (size_t g = 0; g < n_chunks_; ++g) {
            const Ciphertext& ct_x = ct_xs[g];

            // Baby steps over the chunk's input ciphertext.
            std::vector<Ciphertext> baby(n1_);
            baby[0] = ct_x;
            for (size_t j = 1; j < n1_; ++j) {
                int bstep = static_cast<int>((j * B_) % rt_.slot_count);
                if (bstep != 0) {
                    ev.rotate_vector(ct_x, bstep, *rt_.galois_keys, baby[j]);
                } else {
                    baby[j] = ct_x;  // identity
                }
            }

            for (size_t i = 0; i < n2_; ++i) {
                Ciphertext inner;
                bool inner_set = false;
                for (size_t j = 0; j < n1_; ++j) {
                    size_t k = i * n1_ + j;
                    if (k >= chunk_dim_) break;
                    Ciphertext term;
                    ev.multiply_plain(baby[j], pt_diag_bsgs_chunked_[g][i][j], term);
                    if (!inner_set) { inner = std::move(term); inner_set = true; }
                    else { ev.add_inplace(inner, term); }
                }
                if (!inner_set) continue;
                if (i > 0) {
                    int gstep = static_cast<int>((i * n1_ * B_) % rt_.slot_count);
                    if (gstep != 0)
                        ev.rotate_vector_inplace(inner, gstep, *rt_.galois_keys);
                    // gstep == 0 means identity; do nothing
                }
                if (!acc_set) { acc = std::move(inner); acc_set = true; }
                else { ev.add_inplace(acc, inner); }
            }
        }

        if (!acc_set)
            throw runtime_error("layer1_matvec: no terms accumulated");

        ev.rescale_to_next_inplace(acc);
        acc.scale() = rt_.scale;
        ev.add_plain_inplace(acc, pt_b1_at_L1_);

        matvec_ms = chrono::duration<double, milli>(clk::now() - t0).count();
        return acc;
    }

    void compute_powers(const Ciphertext& ct_y,
        int max_deg,
        Ciphertext& ct_y2,
        Ciphertext& ct_y4,
        bool& has_y4,
        double& powers_ms) const {
        auto& ev = *rt_.evaluator;
        auto t0 = clk::now();

        ev.square(ct_y, ct_y2);
        ev.relinearize_inplace(ct_y2, *rt_.relin_keys);
        ev.rescale_to_next_inplace(ct_y2);
        ct_y2.scale() = rt_.scale;

        has_y4 = (max_deg >= 5);
        if (has_y4) {
            ev.square(ct_y2, ct_y4);
            ev.relinearize_inplace(ct_y4, *rt_.relin_keys);
            ev.rescale_to_next_inplace(ct_y4);
            ct_y4.scale() = rt_.scale;
        }
        powers_ms = chrono::duration<double, milli>(clk::now() - t0).count();
    }

    Ciphertext activation(const Ciphertext& y,
        const Ciphertext& y2,
        const Ciphertext& y4,
        const vector<double>& c,
        double& activation_ms) const {
        auto t0 = clk::now();
        Ciphertext ct_p = poly_.eval(y, y2, y4, c);
        activation_ms = chrono::duration<double, milli>(clk::now() - t0).count();
        return ct_p;
    }

    vector<Ciphertext> output_layer_multiclass(const Ciphertext& ct_p,
        double& output_ms) const {
        auto& ev = *rt_.evaluator;
        auto t0 = clk::now();

        const size_t C = meta_.num_classes;
        const size_t H = meta_.hidden_dim;

        auto it = pt_W2_cache_.find(ct_p.parms_id());
        if (it == pt_W2_cache_.end())
            throw runtime_error("output_layer_multiclass: pt_W2 not "
                "pre-encoded for ct_p's parms_id (unexpected post-activation "
                "level)");
        const vector<Plaintext>& pt_W2 = it->second;

        vector<Ciphertext> outs(C);
        for (size_t c = 0; c < C; ++c) {
            Ciphertext ct_s;
            ev.multiply_plain(ct_p, pt_W2[c], ct_s);
            ev.rescale_to_next_inplace(ct_s);
            ct_s.scale() = rt_.scale;

            for (size_t s = H / 2; s >= 1; s /= 2) {
                int rstep = static_cast<int>((s * B_) % rt_.slot_count);
                if (rstep != 0) {
                    Ciphertext rot;
                    ev.rotate_vector(ct_s, rstep, *rt_.galois_keys, rot);
                    ev.add_inplace(ct_s, rot);
                }
                if (s == 1) break;
            }

            if (b2_[c] != 0.0) {
                Plaintext pt_b2c = rt_.encode_scalar_at(b2_[c], ct_s.parms_id());
                ev.add_plain_inplace(ct_s, pt_b2c);
            }
            outs[c] = std::move(ct_s);
        }
        output_ms = chrono::duration<double, milli>(clk::now() - t0).count();
        return outs;
    }

    vector<vector<double>> decrypt_batch_logits_multiclass(
        const vector<Ciphertext>& cts, double& decrypt_ms) const
    {
        auto t0 = clk::now();
        const size_t C = cts.size();
        vector<vector<double>> per_class(C, vector<double>(B_, 0.0));
        for (size_t c = 0; c < C; ++c) {
            Plaintext pt;
            rt_.decryptor->decrypt(cts[c], pt);
            vector<double> v;
            rt_.encoder->decode(pt, v);
            for (size_t b = 0; b < B_; ++b) per_class[c][b] = v[b];
        }
        decrypt_ms = chrono::duration<double, milli>(clk::now() - t0).count();
        return per_class;
    }

    vector<double> plain_forward_multiclass(const vector<double>& x,
        const vector<double>& c) const {
        const size_t d = meta_.input_dim;
        const size_t H = meta_.hidden_dim;
        const size_t C = meta_.num_classes;
        const int    deg = static_cast<int>(c.size()) - 1;

        vector<double> p_vec(H);
        for (size_t i = 0; i < H; ++i) {
            double y = b1_[i];
            for (size_t j = 0; j < d; ++j) y += W1_[i * d + j] * x[j];
            double p = 0.0, ypow = 1.0;
            for (int k = 0; k <= deg; ++k) { p += c[k] * ypow; ypow *= y; }
            p_vec[i] = p;
        }

        vector<double> scores(C);
        for (size_t cl = 0; cl < C; ++cl) {
            double s = b2_[cl];
            for (size_t i = 0; i < H; ++i)
                s += W2_[cl * H + i] * p_vec[i];
            scores[cl] = s;
        }
        return scores;
    }

private:
    void precompute_layer_plaintexts(const std::set<int>& post_act_levels) {
        const size_t H = meta_.hidden_dim;

        auto chain = rt_.build_level_chain();

        // Per-chunk diagonal plaintexts. Each chunk is a (hidden_dim x chunk_dim_)
        // sliced matvec problem with chunk_dim_ == hidden_dim (case_b) or
        // chunk_dim_ == input_dim (case_a, single chunk).
        auto L0 = chain.at(0);
        pt_diag_bsgs_chunked_.assign(n_chunks_,
            std::vector<std::vector<Plaintext>>(n2_, std::vector<Plaintext>(n1_)));

        for (size_t g = 0; g < n_chunks_; ++g) {
            const size_t chunk_offset = is_case_b_ ? g * chunk_dim_ : 0;
            for (size_t i = 0; i < n2_; ++i) {
                for (size_t j = 0; j < n1_; ++j) {
                    size_t k = i * n1_ + j;
                    vector<double> diag(rt_.slot_count, 0.0);
                    if (k < chunk_dim_) {
                        diag = build_diagonal_batched_chunk(
                            W1_, H, meta_.input_dim,
                            chunk_offset, chunk_dim_,
                            k, B_, rt_.slot_count);
                        if (i > 0) {
                            long long shift =
                                -static_cast<long long>(i * n1_ * B_);
                            diag = plaintext_rotate(diag, shift, rt_.slot_count);
                        }
                    }
                    rt_.encoder->encode(diag, L0, rt_.scale,
                                        pt_diag_bsgs_chunked_[g][i][j]);
                }
            }
        }

        auto L1 = chain.at(1);
        vector<double> b1_padded = pad_per_row_batched(b1_, B_, rt_.slot_count);
        rt_.encoder->encode(b1_padded, L1, rt_.scale, pt_b1_at_L1_);

        for (int post_level : post_act_levels) {
            if (post_level <= 0 || (size_t)post_level >= chain.size())
                throw runtime_error("Invalid post-activation level "
                    + to_string(post_level) + " for W2 pre-encoding");
            auto pid = chain[post_level];

            vector<Plaintext> pts(meta_.num_classes);
            for (size_t c = 0; c < meta_.num_classes; ++c) {
                vector<double> w2c(rt_.slot_count, 0.0);
                for (size_t i = 0; i < H; ++i) {
                    const double w = W2_[c * H + i];
                    const size_t base = i * B_;
                    for (size_t b = 0; b < B_; ++b) w2c[base + b] = w;
                }
                rt_.encoder->encode(w2c, pid, rt_.scale, pts[c]);
            }
            pt_W2_cache_[pid] = std::move(pts);
        }

        cout << "[Engine] Pre-encoded W2 at " << pt_W2_cache_.size()
            << " level(s); diagonals: "
            << (n_chunks_ * n1_ * n2_) << " plaintexts ("
            << n_chunks_ << " chunks x " << n1_ << " x " << n2_ << ").\n";
    }

    HECKKSRuntime& rt_;
    ModelMeta            meta_;
    vector<double>       W1_, b1_;
    vector<double>       W2_;
    vector<double>       b2_;

    // Chunked diagonal plaintexts: [g][i][j]
    std::vector<std::vector<std::vector<Plaintext>>> pt_diag_bsgs_chunked_;

    Plaintext            pt_b1_at_L1_;
    std::unordered_map<parms_id_type, std::vector<Plaintext>, ParmsIdHasher>
        pt_W2_cache_;
    PolyEvaluator        poly_;

    size_t B_ = 1;
    size_t n1_ = 1;
    size_t n2_ = 1;

    // Chunk geometry (case_a => n_chunks_=1, chunk_dim_=input_dim;
    //                 case_b => n_chunks_=input_dim/hidden_dim,
    //                           chunk_dim_=hidden_dim)
    size_t n_chunks_  = 1;
    size_t chunk_dim_ = 1;
    bool   is_case_b_ = false;
};

// =============================================================================
// Helpers
// =============================================================================

static int argmax_vec(const double* p, size_t C) {
    int best = 0;
    double bv = p[0];
    for (size_t c = 1; c < C; ++c) {
        if (p[c] > bv) { bv = p[c]; best = static_cast<int>(c); }
    }
    return best;
}

static void banner(const string& s) {
    cout << "\n" << string(78, '=') << "\n" << s << "\n" << string(78, '=') << "\n";
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    // CLI parsing
    string artifact_dir = "he_artifacts_otto_h256";
    if (argc >= 2) artifact_dir = argv[1];
    size_t max_samples = 0;
    if (argc >= 3) max_samples = static_cast<size_t>(stoll(argv[2]));

    unsigned int n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 4;
    if (argc >= 4) n_threads = static_cast<unsigned>(std::max(1, atoi(argv[3])));

    int scale_log2 = 40;
    std::vector<int> coeff_bits = {60, 40, 40, 40, 40, 40, 60};
    std::string output_tag;
    if (argc >= 5) scale_log2 = atoi(argv[4]);
    if (argc >= 6) {
        coeff_bits.clear();
        std::string s = argv[5];
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (!item.empty()) coeff_bits.push_back(atoi(item.c_str()));
        }
    }
    output_tag = (argc >= 7) ? argv[6] : ("s" + std::to_string(scale_log2));
    size_t poly_degree = 16384;
    if (argc >= 8) poly_degree = static_cast<size_t>(stoll(argv[7]));

    banner("Encrypted inference (BATCHED + BSGS, multiclass, multi-threaded): Otto Group");
    cout << "Artifact dir : " << artifact_dir << "\n";
    cout << "Output tag   : " << output_tag << "\n";
    cout << "Poly degree  : " << poly_degree << "\n";
    cout << "Threads      : " << n_threads
        << "  (hardware_concurrency=" << std::thread::hardware_concurrency() << ")\n";

    // Load metadata
    ModelMeta meta = load_meta(artifact_dir + "/meta.txt");
    const size_t orig_input_dim = meta.input_dim;
    cout << "Model        : input_dim=" << meta.input_dim
        << "  hidden_dim=" << meta.hidden_dim
        << "  num_classes=" << meta.num_classes
        << "  n_test=" << meta.n_test
        << "  n_schemes=" << meta.n_schemes << "\n";

    // Load model tensors
    auto W1 = load_bin<double>(artifact_dir + "/W1.bin",
        meta.hidden_dim * orig_input_dim);
    auto b1 = load_bin<double>(artifact_dir + "/b1.bin", meta.hidden_dim);
    auto W2 = load_bin<double>(artifact_dir + "/W2.bin",
        meta.num_classes * meta.hidden_dim);
    auto b2 = load_bin<double>(artifact_dir + "/b2.bin", meta.num_classes);

    auto X_test = load_bin<double>(artifact_dir + "/X_test.bin",
        meta.n_test * orig_input_dim);
    auto y_test_true = load_bin<int32_t>(artifact_dir + "/y_test_true.bin", meta.n_test);
    auto y_test_relu = load_bin<int32_t>(artifact_dir + "/y_test_relu.bin", meta.n_test);

    // BSGS cyclic correctness requires either:
    //   (a) hidden_dim % input_dim == 0     (input_dim <  hidden_dim, e.g. Otto)
    //   (b) input_dim  % hidden_dim == 0    (input_dim >= hidden_dim, e.g. DINOv2)
    // We pad input_dim to satisfy whichever applies.
    size_t input_dim_padded = orig_input_dim;
    if (orig_input_dim < meta.hidden_dim) {
        // case (a): pad up so hidden_dim is a multiple of input_dim
        while (input_dim_padded <= meta.hidden_dim &&
               meta.hidden_dim % input_dim_padded != 0) {
            ++input_dim_padded;
        }
        if (meta.hidden_dim % input_dim_padded != 0)
            throw runtime_error("Cannot pad input_dim to a divisor of hidden_dim");
    } else {
        // case (b): pad up so input_dim is a multiple of hidden_dim
        while (input_dim_padded % meta.hidden_dim != 0) {
            ++input_dim_padded;
        }
    }

    if (input_dim_padded != orig_input_dim) {
        cout << "[Pad] input_dim " << orig_input_dim << " -> " << input_dim_padded
             << " (BSGS cyclic-correctness requires "
             << (orig_input_dim < meta.hidden_dim ? "hidden_dim" : "input_dim")
             << " % "
             << (orig_input_dim < meta.hidden_dim ? "input_dim" : "hidden_dim")
             << " == 0)\n";

        vector<double> W1_p(meta.hidden_dim * input_dim_padded, 0.0);
        for (size_t i = 0; i < meta.hidden_dim; ++i)
            for (size_t j = 0; j < orig_input_dim; ++j)
                W1_p[i * input_dim_padded + j] = W1[i * orig_input_dim + j];
        W1 = std::move(W1_p);

        vector<double> X_p(meta.n_test * input_dim_padded, 0.0);
        for (size_t i = 0; i < meta.n_test; ++i)
            for (size_t j = 0; j < orig_input_dim; ++j)
                X_p[i * input_dim_padded + j] = X_test[i * orig_input_dim + j];
        X_test = std::move(X_p);

        meta.input_dim = input_dim_padded;
    }

    // Load schemes
    auto scheme_cfgs = load_schemes(artifact_dir + "/schemes.txt");
    const size_t loaded_count = scheme_cfgs.size();
    scheme_cfgs.erase(
        std::remove_if(scheme_cfgs.begin(), scheme_cfgs.end(),
            [](const SchemeConfig& c) { return scheme_is_skipped(c.name); }),
        scheme_cfgs.end());
    if (scheme_cfgs.empty())
        throw runtime_error("No schemes left after filtering");
    cout << "Schemes      : " << scheme_cfgs.size() << " kept ("
        << (loaded_count - scheme_cfgs.size()) << " filtered out: ";
    for (size_t i = 0; i < SKIP_SCHEMES.size(); ++i)
        cout << (i ? ", " : "") << SKIP_SCHEMES[i];
    cout << ")\n";

    vector<SchemeResults> schemes;
    schemes.reserve(scheme_cfgs.size());
    int max_degree = 0;
    std::set<int> post_act_levels;
    for (const auto& cfg : scheme_cfgs) {
        SchemeResults r;
        r.name = cfg.name;
        r.degree = cfg.degree;
        r.activation_depth = poly_activation_depth(cfg.degree);
        r.total_circuit_depth = 1 + r.activation_depth + 1;
        post_act_levels.insert(1 + r.activation_depth);
        r.coeffs = load_bin<double>(artifact_dir + "/coeffs_" + cfg.name + ".bin",
            cfg.degree + 1);
        r.plain_logits = load_bin<double>(
            artifact_dir + "/logits_" + cfg.name + ".bin",
            meta.n_test * meta.num_classes);
        schemes.push_back(std::move(r));
        max_degree = std::max(max_degree, cfg.degree);
    }

    cout << "Schemes details:\n";
    for (const auto& s : schemes) {
        cout << "   - " << setw(12) << left << s.name << right
            << "  deg=" << s.degree
            << "  act_depth=" << s.activation_depth
            << "  total_depth=" << s.total_circuit_depth
            << "  coeffs(asc)=[";
        for (size_t k = 0; k < s.coeffs.size(); ++k)
            cout << (k ? ", " : "") << s.coeffs[k];
        cout << "]\n";
    }

    // BSGS factorization. The factorization target is chunk_dim, which equals
    // hidden_dim in case_b (input_dim > hidden_dim) and input_dim in case_a.
    const size_t slot_count_for_keys = poly_degree / 2;
    if (slot_count_for_keys % meta.hidden_dim != 0)
        throw runtime_error("slot_count must be divisible by hidden_dim");
    const size_t B = slot_count_for_keys / meta.hidden_dim;

    const bool case_b_main =
        (meta.input_dim > meta.hidden_dim) &&
        (meta.input_dim % meta.hidden_dim == 0);
    const size_t bsgs_dim = case_b_main ? meta.hidden_dim : meta.input_dim;
    const size_t n_chunks_main = case_b_main
        ? (meta.input_dim / meta.hidden_dim) : 1;

    size_t n1 = static_cast<size_t>(std::ceil(std::sqrt(static_cast<double>(bsgs_dim))));
    while (n1 * ((bsgs_dim + n1 - 1) / n1) < bsgs_dim) ++n1;
    size_t n2 = (bsgs_dim + n1 - 1) / n1;
    cout << "Batch        : B=" << B << "   BSGS: n1=" << n1 << "  n2=" << n2
        << "   bsgs_dim=" << bsgs_dim
        << "   n_chunks=" << n_chunks_main
        << "   (input_dim=" << meta.input_dim
        << ", hidden_dim=" << meta.hidden_dim << ")\n";

    auto rotation_steps = InferenceEngine::compute_rotation_steps(
        B, meta.hidden_dim, n1, n2, slot_count_for_keys);
    cout << "Rotation steps used (" << rotation_steps.size() << "): ";
    for (auto s : rotation_steps) cout << s << " ";
    cout << "\n";

    // Build SEAL runtime + engine
    banner("Building CKKS context, keys and pre-encoded plaintexts");
    HECKKSRuntime rt(rotation_steps, scale_log2, coeff_bits, poly_degree);

    int max_total_depth = 0;
    for (const auto& s : schemes)
        max_total_depth = std::max(max_total_depth, s.total_circuit_depth);
    bool bootstrap_needed =
        static_cast<size_t>(max_total_depth) > rt.max_rescales_available;
    cout << "Max circuit depth across schemes : " << max_total_depth << "\n"
        << "Rescales available in chain      : " << rt.max_rescales_available << "\n"
        << "Bootstrapping needed             : "
        << (bootstrap_needed ? "YES" : "NO") << "\n";

    // Auto-filter infeasible schemes if depth budget is insufficient
    if (bootstrap_needed) {
        cout << "\n[Filter] Some schemes exceed available depth. Removing them...\n";
        const size_t before = schemes.size();
        schemes.erase(
            std::remove_if(schemes.begin(), schemes.end(),
                [&](const SchemeResults& s) {
                    bool infeasible = static_cast<size_t>(s.total_circuit_depth)
                                    > rt.max_rescales_available;
                    if (infeasible) {
                        cout << "  - Skipped: " << s.name
                             << " (depth=" << s.total_circuit_depth
                             << " > budget=" << rt.max_rescales_available << ")\n";
                    }
                    return infeasible;
                }),
            schemes.end());
        cout << "  Kept " << schemes.size() << "/" << before
             << " feasible schemes for this configuration.\n";
        if (schemes.empty()) {
            cout << "[Error] No feasible schemes remain. Exiting.\n";
            return 1;
        }
        // Recompute max_degree & post_act_levels because we removed schemes
        max_degree = 0;
        post_act_levels.clear();
        for (const auto& s : schemes) {
            max_degree = std::max(max_degree, s.degree);
            post_act_levels.insert(1 + s.activation_depth);
        }
    }

    // Build the engine (uses the (possibly filtered) post_act_levels)
    InferenceEngine engine(rt, meta, W1, b1, W2, b2, n1, n2, post_act_levels);

    // Multi-threaded inference loop
    banner("Running encrypted inference on the test set (parallel)");
    const size_t N = (max_samples == 0 || max_samples > meta.n_test)
        ? meta.n_test : max_samples;
    const size_t n_batches = (N + B - 1) / B;
    const size_t C = meta.num_classes;
    cout << fixed << setprecision(6);
    cout << "Total samples=" << N << "  batches=" << n_batches
        << "  (B=" << B << " per batch, C=" << C
        << ", " << n_threads << " threads)\n";

    print_operation_counts(schemes, meta, B, rt.slot_count, n1, n2, max_degree);

    vector<vector<SchemeAcc>> per_thread_acc(
        n_threads, vector<SchemeAcc>(schemes.size()));

    // Per-sample HE logits (shared among threads; disjoint writes are safe)
    vector<vector<double>> he_logits_full(
        schemes.size(), vector<double>(N * C, 0.0));

    std::atomic<size_t> next_batch{ 0 };
    std::atomic<size_t> batches_done{ 0 };

    std::vector<DebugSample> debug_samples;
    std::mutex debug_mutex;
    std::mutex progress_mutex;

    const size_t progress_every = std::max<size_t>(1, n_batches / 20);
    const auto loop_start = clk::now();

    auto worker = [&](unsigned tid) {
        size_t bi;
        while ((bi = next_batch.fetch_add(1, std::memory_order_relaxed))
            < n_batches)
        {
            const size_t start = bi * B;
            const size_t bsize = std::min(B, N - start);

            vector<vector<double>> xs(bsize, vector<double>(meta.input_dim));
            for (size_t b = 0; b < bsize; ++b)
                for (size_t j = 0; j < meta.input_dim; ++j)
                    xs[b][j] = X_test[(start + b) * meta.input_dim + j];

            double batch_t_enc = 0.0;
            vector<Ciphertext> ct_xs = engine.encrypt_input_batched(xs, batch_t_enc);

            double batch_t_matvec = 0.0;
            Ciphertext ct_y = engine.layer1_matvec(ct_xs, batch_t_matvec);

            Ciphertext ct_y2, ct_y4;
            bool has_y4 = false;
            double batch_t_powers = 0.0;
            engine.compute_powers(ct_y, max_degree,
                ct_y2, ct_y4, has_y4, batch_t_powers);

            vector<vector<vector<double>>> he_logits_per_scheme(
                schemes.size(),
                vector<vector<double>>(bsize, vector<double>(C, 0.0)));

            vector<double> per_scheme_t_act(schemes.size(), 0.0);
            vector<double> per_scheme_t_out(schemes.size(), 0.0);
            vector<double> per_scheme_t_dec(schemes.size(), 0.0);

            for (size_t si = 0; si < schemes.size(); ++si) {
                const auto& s = schemes[si];
                double t_act = 0.0, t_out = 0.0, t_dec = 0.0;

                Ciphertext ct_p = engine.activation(
                    ct_y, ct_y2, has_y4 ? ct_y4 : ct_y2, s.coeffs, t_act);

                vector<Ciphertext> ct_scores =
                    engine.output_layer_multiclass(ct_p, t_out);

                vector<vector<double>> per_class =
                    engine.decrypt_batch_logits_multiclass(ct_scores, t_dec);

                for (size_t b = 0; b < bsize; ++b)
                    for (size_t c = 0; c < C; ++c)
                        he_logits_per_scheme[si][b][c] = per_class[c][b];

                per_scheme_t_act[si] = t_act;
                per_scheme_t_out[si] = t_out;
                per_scheme_t_dec[si] = t_dec;
            }

            for (size_t b = 0; b < bsize; ++b) {
                const size_t idx = start + b;

                for (size_t si = 0; si < schemes.size(); ++si) {
                    const auto& s = schemes[si];
                    auto& acc = per_thread_acc[tid][si];

                    const double* he_v = he_logits_per_scheme[si][b].data();
                    const double* py_v = &s.plain_logits[idx * C];

                    vector<double> cpp_v =
                        engine.plain_forward_multiclass(xs[b], s.coeffs);

                    double sample_max_plain_cpp_err = 0.0;
                    for (size_t c = 0; c < C; ++c) {
                        double e_pcp = std::fabs(cpp_v[c] - py_v[c]);
                        if (e_pcp > sample_max_plain_cpp_err)
                            sample_max_plain_cpp_err = e_pcp;
                        double e = std::fabs(he_v[c] - py_v[c]);
                        acc.sum_abs_err += e;
                        if (e > acc.max_abs_err) acc.max_abs_err = e;
                        acc.n_logits++;
                    }
                    if (sample_max_plain_cpp_err > acc.max_plain_cpp_err)
                        acc.max_plain_cpp_err = sample_max_plain_cpp_err;

                    int y_true = y_test_true[idx];
                    int y_relu = y_test_relu[idx];
                    int plain_pred = argmax_vec(py_v, C);
                    int he_pred = argmax_vec(he_v, C);

                    acc.n++;
                    if (plain_pred == y_true) acc.plain_correct_vs_true++;
                    if (he_pred == y_true) acc.he_correct_vs_true++;
                    if (plain_pred == y_relu) acc.plain_correct_vs_relu++;
                    if (he_pred == y_relu) acc.he_correct_vs_relu++;
                    if (plain_pred == he_pred) acc.plain_he_agree++;

                    acc.t_enc += batch_t_enc / bsize;
                    acc.t_matvec += batch_t_matvec / bsize;
                    acc.t_powers += batch_t_powers / bsize;
                    acc.t_act += per_scheme_t_act[si] / bsize;
                    acc.t_out += per_scheme_t_out[si] / bsize;
                    acc.t_dec += per_scheme_t_dec[si] / bsize;
                }

                if (idx < 3 || idx == N - 1) {
                    std::stringstream ss;
                    ss << "[sample " << setw(5) << idx
                        << "  y_true=" << y_test_true[idx]
                        << "  y_relu=" << y_test_relu[idx] << "]\n";
                    for (size_t si = 0; si < schemes.size(); ++si) {
                        const auto& s = schemes[si];
                        const double* he_v = he_logits_per_scheme[si][b].data();
                        const double* py_v = &s.plain_logits[idx * C];
                        int he_pred = argmax_vec(he_v, C);
                        int plain_pred = argmax_vec(py_v, C);
                        double max_e = 0.0;
                        for (size_t c = 0; c < C; ++c)
                            max_e = std::max(max_e, std::fabs(he_v[c] - py_v[c]));
                        ss << "    " << left << setw(12) << s.name << right
                            << "  HE_argmax=" << he_pred
                            << "  Py_argmax=" << plain_pred
                            << "  max_c|HE-Py|=" << scientific << setprecision(3)
                            << max_e << fixed << setprecision(6)
                            << "  HE_logits=[";
                        for (size_t c = 0; c < C; ++c)
                            ss << (c ? "," : "") << showpos << setprecision(3) << he_v[c];
                        ss << noshowpos << setprecision(6) << "]\n";
                    }
                    {
                        std::lock_guard<std::mutex> lock(debug_mutex);
                        debug_samples.push_back({ idx, ss.str() });
                    }
                }
            }

            // Save per-sample HE logits to global array (disjoint writes)
            for (size_t si = 0; si < schemes.size(); ++si) {
                for (size_t b = 0; b < bsize; ++b) {
                    const size_t idx = start + b;
                    for (size_t c = 0; c < C; ++c)
                        he_logits_full[si][idx * C + c]
                            = he_logits_per_scheme[si][b][c];
                }
            }

            size_t done = batches_done.fetch_add(1, std::memory_order_relaxed) + 1;
            if (done < n_batches && done % progress_every == 0) {
                std::lock_guard<std::mutex> lock(progress_mutex);
                double elapsed_s = chrono::duration<double>(
                    clk::now() - loop_start).count();
                double rate_bps = elapsed_s > 0 ? double(done) / elapsed_s : 0.0;
                size_t samples_done = std::min(N, done * B);
                double eta_s = rate_bps > 0
                    ? (double(n_batches - done) / rate_bps) : 0.0;
                cout << "  ... " << setw(6) << samples_done << "/" << N
                    << "   " << setprecision(1) << rate_bps * B << " samples/s"
                    << "   ETA " << eta_s << " s\n"
                    << setprecision(6);
                cout.flush();
            }
        }
    };

    vector<std::thread> workers;
    workers.reserve(n_threads);
    for (unsigned t = 0; t < n_threads; ++t) workers.emplace_back(worker, t);
    for (auto& w : workers) w.join();

    const auto loop_end = clk::now();
    const double wall_clock_s =
        chrono::duration<double>(loop_end - loop_start).count();

    // Print sorted debug samples
    std::sort(debug_samples.begin(), debug_samples.end(),
        [](const DebugSample& a, const DebugSample& b) { return a.idx < b.idx; });
    cout << "\n--- Debug samples (idx 0..2 and N-1) ---\n";
    for (const auto& ds : debug_samples) cout << ds.text;

    // Merge per-thread accumulators
    for (unsigned t = 0; t < n_threads; ++t) {
        for (size_t si = 0; si < schemes.size(); ++si) {
            const auto& src = per_thread_acc[t][si];
            auto& dst = schemes[si];
            dst.n += src.n;
            dst.plain_correct_vs_true += src.plain_correct_vs_true;
            dst.he_correct_vs_true += src.he_correct_vs_true;
            dst.plain_correct_vs_relu += src.plain_correct_vs_relu;
            dst.he_correct_vs_relu += src.he_correct_vs_relu;
            dst.plain_he_agree += src.plain_he_agree;
            dst.sum_abs_err += src.sum_abs_err;
            dst.max_abs_err = std::max(dst.max_abs_err, src.max_abs_err);
            dst.max_plain_cpp_err = std::max(dst.max_plain_cpp_err,
                src.max_plain_cpp_err);
            dst.n_logits += src.n_logits;
            dst.t_enc += src.t_enc;
            dst.t_matvec += src.t_matvec;
            dst.t_powers += src.t_powers;
            dst.t_act += src.t_act;
            dst.t_out += src.t_out;
            dst.t_dec += src.t_dec;
        }
    }

    // Reporting
    banner("Metric summary (multiclass argmax, " + to_string(C) + " classes)");
    cout << fixed;

    cout << setprecision(4);
    cout << "Vs ground truth (y_test_true):\n";
    cout << "  " << left << setw(14) << "Scheme" << right
        << setw(12) << "Plain-ACC"
        << setw(12) << "HE-ACC"
        << setw(16) << "Plain<->HE"
        << setw(12) << "PH-mis"
        << setw(14) << "MaxLogitErr"
        << setw(14) << "MeanLogitErr" << "\n";
    cout << "  " << string(94, '-') << "\n";
    for (const auto& s : schemes) {
        double plain_acc = double(s.plain_correct_vs_true) / s.n;
        double he_acc = double(s.he_correct_vs_true) / s.n;
        double agree = double(s.plain_he_agree) / s.n;
        const size_t plain_he_mismatch = s.n - s.plain_he_agree;
        double mean_err = s.sum_abs_err / std::max<size_t>(1, s.n_logits);
        cout << "  " << left << setw(14) << s.name << right
            << setw(12) << plain_acc
            << setw(12) << he_acc
            << setw(16) << agree
            << setw(12) << plain_he_mismatch
            << scientific << setprecision(3)
            << setw(14) << s.max_abs_err
            << setw(14) << mean_err
            << fixed << setprecision(4) << "\n";
    }

    cout << "\nVs ReLU's decisions (y_test_relu):\n";
    cout << "  " << left << setw(14) << "Scheme" << right
        << setw(16) << "Plain-AgreeReLU"
        << setw(16) << "HE-AgreeReLU" << "\n";
    cout << "  " << string(52, '-') << "\n";
    for (const auto& s : schemes) {
        double p = double(s.plain_correct_vs_relu) / s.n;
        double h = double(s.he_correct_vs_relu) / s.n;
        cout << "  " << left << setw(14) << s.name << right
            << setw(16) << p << setw(16) << h << "\n";
    }

    cout << "\nDepth / bootstrapping:\n";
    cout << "  " << left << setw(14) << "Scheme" << right
        << setw(8) << "Degree"
        << setw(12) << "ActDepth"
        << setw(14) << "TotalDepth"
        << setw(22) << "Bootstrap needed?" << "\n";
    cout << "  " << string(74, '-') << "\n";
    for (const auto& s : schemes) {
        bool bs = static_cast<size_t>(s.total_circuit_depth) > rt.max_rescales_available;
        cout << "  " << left << setw(14) << s.name << right
            << setw(8) << s.degree
            << setw(12) << s.activation_depth
            << setw(14) << s.total_circuit_depth
            << setw(22) << (bs ? "YES" : "NO") << "\n";
    }
    cout << "  (Chain supports " << rt.max_rescales_available
        << " rescales)\n";

    cout << "\nLatency breakdown (avg ms per sample, sequential-equivalent):\n";
    cout << "  " << left << setw(14) << "Scheme" << right
        << setw(10) << "Encrypt"
        << setw(10) << "Matvec"
        << setw(10) << "Powers"
        << setw(10) << "Activ."
        << setw(10) << "Output"
        << setw(10) << "Decrypt"
        << setw(10) << "Server"
        << setw(10) << "Total" << "\n";
    cout << "  " << string(94, '-') << "\n";
    cout << setprecision(3);
    for (const auto& s : schemes) {
        double n = double(s.n);
        double enc = s.t_enc / n;
        double mv = s.t_matvec / n;
        double pw = s.t_powers / n;
        double ac = s.t_act / n;
        double ou = s.t_out / n;
        double de = s.t_dec / n;
        double server = mv + pw + ac + ou;
        double tot = enc + server + de;
        cout << "  " << left << setw(14) << s.name << right
            << setw(10) << enc
            << setw(10) << mv
            << setw(10) << pw
            << setw(10) << ac
            << setw(10) << ou
            << setw(10) << de
            << setw(10) << server
            << setw(10) << tot << "\n";
    }
    cout << "  Note: 'sequential-equivalent' = sum of per-thread per-batch\n"
        << "        time / total samples. Wall-clock speedup from threading\n"
        << "        is reported below.\n";

    {
        double seq_total_s = 0.0;
        for (unsigned t = 0; t < n_threads; ++t) {
            const auto& a = per_thread_acc[t][0];
            seq_total_s += (a.t_enc + a.t_matvec + a.t_powers
                          + a.t_act + a.t_out + a.t_dec) / 1000.0;
        }
        const double speedup = seq_total_s / wall_clock_s;
        const double efficiency = speedup / n_threads;

        cout << "\n[Timing] Wall-clock for parallel inference: "
             << wall_clock_s << " s   ("
             << (double)N / wall_clock_s << " samples/s, "
             << n_threads << " threads)\n"
             << "         Sequential-equivalent total      : "
             << seq_total_s << " s\n"
             << "         Parallel speedup                 : "
             << speedup << "x  (efficiency = "
             << efficiency * 100 << "%)\n";
    }

    cout << "\nPlaintext correctness (C++ forward vs Python export):\n";
    cout << setprecision(3);
    for (const auto& s : schemes) {
        cout << "  " << left << setw(14) << s.name << right
            << "  max_{i,c}|cpp_plain - py_plain| = " << scientific
            << s.max_plain_cpp_err << fixed << "\n";
    }

    // Save HE logits per scheme
    cout << "\n[Save] Writing HE logits to " << artifact_dir << "/ ...\n";
    for (size_t si = 0; si < schemes.size(); ++si) {
        const string fname = artifact_dir + "/he_logits_" + schemes[si].name
                             + "_" + output_tag + ".bin";
        ofstream fout(fname, ios::binary);
        if (!fout) {
            cerr << "[WARN] Cannot open " << fname << " for writing\n";
            continue;
        }
        fout.write(reinterpret_cast<const char*>(he_logits_full[si].data()),
                   static_cast<streamsize>(he_logits_full[si].size() * sizeof(double)));
        cout << "  -> " << fname << "  (" << N << " x " << C << " doubles)\n";
    }

    return 0;
}