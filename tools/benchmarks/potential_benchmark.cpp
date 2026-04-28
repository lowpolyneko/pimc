#include "common.h"
#include "action.h"
#include "container.h"
#include "factory_potential.h"
#include "lookuptable.h"
#include "path.h"
#include "potential.h"
#include "setup.h"
#include "wavefunction.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>

namespace {

struct BenchmarkOptions {
    std::string kind = "interaction";
    std::string potential = "free";
    std::string method = "all";
    std::size_t iterations = 1000000;
    std::size_t samples = 4096;
    double minRadius = 0.25;
    double maxRadius = 5.0;
    bool gpu = false;
    bool checkBatch = false;
    bool checkGpu = false;
    bool list = false;
    bool help = false;
};

bool isBenchmarkOption(const std::string& arg, const std::string& name)
{
    return arg == name || arg.rfind(name + "=", 0) == 0;
}

std::string optionValue(const std::string& arg, const std::string& name,
        int& i, int argc, char* argv[])
{
    const std::string prefix = name + "=";
    if (arg.rfind(prefix, 0) == 0)
        return arg.substr(prefix.size());
    if (i + 1 >= argc)
        throw std::runtime_error("missing value for " + name);
    return argv[++i];
}

bool hasOption(const std::vector<std::string>& args,
        const std::string& longName, const std::string& shortName = "")
{
    for (const auto& arg : args) {
        if (arg == longName || arg.rfind(longName + "=", 0) == 0)
            return true;
        if (!shortName.empty() && arg == shortName)
            return true;
    }
    return false;
}

void appendDefault(std::vector<std::string>& args,
        const std::string& longName, const std::string& value,
        const std::string& shortName = "")
{
    if (!hasOption(args, longName, shortName)) {
        args.push_back(longName);
        args.push_back(value);
    }
}

void printNames(const std::string& label, const std::vector<std::string>& names)
{
    std::cout << label << " potentials:" << std::endl;
    for (const auto& name : names)
        std::cout << "  " << name << std::endl;
}

void printHelp(const char* exe)
{
    std::cout
        << "Usage: " << exe << " --benchmark-kind interaction|external"
        << " --benchmark-potential NAME [benchmark options] [PIMC options]\n\n"
        << "Benchmark options:\n"
        << "  --benchmark-list                 list registered potentials\n"
        << "  --benchmark-kind KIND            interaction or external\n"
        << "  --benchmark-potential NAME       potential to instantiate\n"
        << "  --benchmark-method METHOD        value, gradient, pair, action, or all\n"
        << "  --benchmark-iterations N         calls per measured method [1000000]\n"
        << "  --benchmark-samples N            generated sample positions [4096]\n"
        << "  --benchmark-min R                minimum sample radius [0.25]\n"
        << "  --benchmark-max R                maximum sample radius [5.0]\n\n"
        << "  --benchmark-gpu                  use GPPotential batched GPU value path\n\n"
        << "  --benchmark-check-batch          compare public batched V calls to scalar V\n"
        << "  --benchmark-check-gpu            compare GPPotential batched V, using GPU build, to scalar V\n"
        << "  --benchmark-tolerance X          relative/absolute check tolerance [1e-9]\n\n"
        << "Any remaining options are passed through the normal PIMC setup parser.\n"
        << "The benchmark supplies defaults for size, temperature, particle count,\n"
        << "time slices, tube radius, and common LJ/Gasparini parameters.\n";
}

dVec sampleVector(std::size_t i, std::size_t n, double minRadius,
        double maxRadius)
{
    dVec r{};
    const double t = (n > 1) ? static_cast<double>(i % n) / (n - 1) : 0.0;
    const double radius = minRadius + (maxRadius - minRadius) * t;
    for (int d = 0; d < NDIM; ++d) {
        const double phase = 0.731 * static_cast<double>(i + 1) *
            static_cast<double>(d + 1);
        r[d] = radius * std::sin(phase);
    }
    if (dot(r, r) < EPS)
        r[0] = minRadius;
    return r;
}

std::vector<dVec> makeSamples(std::size_t count, double minRadius,
        double maxRadius)
{
    std::vector<dVec> samples;
    samples.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
        samples.push_back(sampleVector(i, count, minRadius, maxRadius));
    return samples;
}

void appendGpuCheckSamples(std::vector<dVec>& samples)
{
#if NDIM > 2
    const double coords[][NDIM] = {
        {0.1, 0.2, 1.0},
        {3.1, 0.0, 1.9},
        {-4.0, 2.5, -3.0},
        {5.5, 0.25, 6.75},
        {-6.0, -3.0, 7.5},
        {0.0, 6.5, -6.6},
    };

    for (const auto& coord : coords) {
        dVec r{};
        for (int d = 0; d < NDIM; ++d)
            r[d] = coord[d];
        samples.push_back(r);
    }
#else
    (void)samples;
#endif
}

std::vector<double> flattenSamples(const std::vector<dVec>& samples)
{
    std::vector<double> flatSamples(samples.size() * NDIM);
    for (std::size_t i = 0; i < samples.size(); ++i) {
        for (int d = 0; d < NDIM; ++d)
            flatSamples[NDIM * i + d] = samples[i][d];
    }
    return flatSamples;
}

template <typename Func>
void runTimed(const std::string& label, std::size_t iterations, Func func)
{
    volatile double sink = 0.0;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iterations; ++i)
        sink += func(i);
    const auto stop = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = stop - start;
    const double seconds = elapsed.count();
    const double nsPerCall = seconds * 1.0e9 / static_cast<double>(iterations);

    std::cout << std::setw(12) << label
              << " calls=" << iterations
              << " seconds=" << std::setprecision(8) << seconds
              << " ns_per_call=" << std::setprecision(8) << nsPerCall
              << " checksum=" << std::setprecision(12) << sink
              << std::endl;
}

double comparisonScale(double expected)
{
    return std::max(1.0, std::fabs(expected));
}

int checkBatchedValues(PotentialBase& potential, const std::vector<dVec>& samples,
        double tolerance)
{
    std::vector<double> expected(samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i)
        expected[i] = potential.V(samples[i]);

    std::vector<double> actual(samples.size());
    potential.V(samples.data(), actual.data(), static_cast<int>(samples.size()));

    double maxAbsDiff = 0.0;
    double maxRelDiff = 0.0;
    std::size_t worst = 0;
    for (std::size_t i = 0; i < samples.size(); ++i) {
        const double absDiff = std::fabs(actual[i] - expected[i]);
        const double relDiff = absDiff / comparisonScale(expected[i]);
        if (relDiff > maxRelDiff) {
            maxRelDiff = relDiff;
            maxAbsDiff = absDiff;
            worst = i;
        }
        if (absDiff > tolerance && relDiff > tolerance) {
            std::cerr << "error: batched V mismatch at sample " << i
                      << " expected=" << std::setprecision(17) << expected[i]
                      << " actual=" << std::setprecision(17) << actual[i]
                      << " abs_diff=" << absDiff
                      << " rel_diff=" << relDiff
                      << std::endl;
            return EXIT_FAILURE;
        }
    }

    const std::size_t batchSizes[] = {1, 2, 3, 7, 16, 64, 257};
    std::vector<double> batchActual(samples.size());
    for (const auto batchSize : batchSizes) {
        for (std::size_t begin = 0; begin < samples.size(); begin += batchSize) {
            const std::size_t n = std::min(batchSize, samples.size() - begin);
            potential.V(samples.data() + begin, batchActual.data() + begin,
                    static_cast<int>(n));
        }
        for (std::size_t i = 0; i < samples.size(); ++i) {
            const double absDiff = std::fabs(batchActual[i] - expected[i]);
            const double relDiff = absDiff / comparisonScale(expected[i]);
            if (absDiff > tolerance && relDiff > tolerance) {
                std::cerr << "error: batched V mismatch with batch_size=" << batchSize
                          << " at sample " << i
                          << " expected=" << std::setprecision(17) << expected[i]
                          << " actual=" << std::setprecision(17) << batchActual[i]
                          << " abs_diff=" << absDiff
                          << " rel_diff=" << relDiff
                          << std::endl;
                return EXIT_FAILURE;
            }
        }
    }

    std::cout << "batch correctness check passed"
              << " samples=" << samples.size()
              << " tolerance=" << std::setprecision(3) << tolerance
              << " max_abs_diff=" << std::setprecision(12) << maxAbsDiff
              << " max_rel_diff=" << std::setprecision(12) << maxRelDiff
              << " worst_sample=" << worst
              << std::endl;
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char* argv[])
{
    BenchmarkOptions bench;
    double checkTolerance = 1.0e-9;
    std::vector<std::string> setupArgs;
    setupArgs.push_back(argv[0]);

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);

            if (arg == "--benchmark-help") {
                bench.help = true;
            } else if (arg == "--benchmark-list") {
                bench.list = true;
            } else if (arg == "--benchmark-gpu") {
                bench.gpu = true;
            } else if (arg == "--benchmark-check-batch") {
                bench.checkBatch = true;
            } else if (arg == "--benchmark-check-gpu") {
                bench.checkGpu = true;
                bench.checkBatch = true;
                bench.gpu = true;
            } else if (isBenchmarkOption(arg, "--benchmark-kind")) {
                bench.kind = optionValue(arg, "--benchmark-kind", i, argc, argv);
            } else if (isBenchmarkOption(arg, "--benchmark-potential")) {
                bench.potential = optionValue(arg, "--benchmark-potential", i, argc, argv);
            } else if (isBenchmarkOption(arg, "--benchmark-method")) {
                bench.method = optionValue(arg, "--benchmark-method", i, argc, argv);
            } else if (isBenchmarkOption(arg, "--benchmark-iterations")) {
                bench.iterations = std::stoull(optionValue(arg, "--benchmark-iterations", i, argc, argv));
            } else if (isBenchmarkOption(arg, "--benchmark-samples")) {
                bench.samples = std::stoull(optionValue(arg, "--benchmark-samples", i, argc, argv));
            } else if (isBenchmarkOption(arg, "--benchmark-min")) {
                bench.minRadius = std::stod(optionValue(arg, "--benchmark-min", i, argc, argv));
            } else if (isBenchmarkOption(arg, "--benchmark-max")) {
                bench.maxRadius = std::stod(optionValue(arg, "--benchmark-max", i, argc, argv));
            } else if (isBenchmarkOption(arg, "--benchmark-tolerance")) {
                checkTolerance = std::stod(optionValue(arg, "--benchmark-tolerance", i, argc, argv));
            } else {
                setupArgs.push_back(arg);
            }
        }
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    if (bench.help) {
        printHelp(argv[0]);
        return EXIT_SUCCESS;
    }

    if (bench.list) {
        printNames("Interaction",
                PotentialFactory::instance().getNames<PotentialFactory::Type::Interaction>());
        printNames("External",
                PotentialFactory::instance().getNames<PotentialFactory::Type::External>());
        return EXIT_SUCCESS;
    }

    if (bench.kind != "interaction" && bench.kind != "external") {
        std::cerr << "error: --benchmark-kind must be interaction or external" << std::endl;
        return EXIT_FAILURE;
    }
    if (bench.method != "value" && bench.method != "gradient" &&
            bench.method != "pair" && bench.method != "action" &&
            bench.method != "all") {
        std::cerr << "error: --benchmark-method must be value, gradient, pair, action, or all" << std::endl;
        return EXIT_FAILURE;
    }
    if (bench.iterations == 0 || bench.samples == 0 ||
            bench.minRadius <= 0.0 || bench.maxRadius <= bench.minRadius ||
            checkTolerance <= 0.0 || !std::isfinite(checkTolerance)) {
        std::cerr << "error: invalid benchmark sampling parameters" << std::endl;
        return EXIT_FAILURE;
    }

    appendDefault(setupArgs, "--temperature", "1.0", "-T");
    appendDefault(setupArgs, "--number_particles", "16", "-N");
    appendDefault(setupArgs, "--size", "10.0", "-L");
    appendDefault(setupArgs, "--number_time_slices", "10", "-P");
    appendDefault(setupArgs, "--action", "pair_product");
    appendDefault(setupArgs, "--radius", "5.0", "-r");
    appendDefault(setupArgs, "--lj_width", "3.0");
    appendDefault(setupArgs, "--lj_density", "0.021");
    appendDefault(setupArgs, "--empty_width_y", "1.0", "-y");
    appendDefault(setupArgs, "--empty_width_z", "1.0", "-z");

    if (bench.kind == "interaction") {
        setupArgs.push_back("--interaction");
        setupArgs.push_back(bench.potential);
        appendDefault(setupArgs, "--external", "free", "-X");
    } else {
        setupArgs.push_back("--external");
        setupArgs.push_back(bench.potential);
        appendDefault(setupArgs, "--interaction", "free", "-I");
    }

    std::vector<char*> setupArgv;
    setupArgv.reserve(setupArgs.size());
    for (auto& arg : setupArgs)
        setupArgv.push_back(arg.data());

    Setup& setup = Setup::instance();
    try {
        setup.getOptions(static_cast<int>(setupArgv.size()), setupArgv.data());
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    if (setup.parseOptions())
        return EXIT_FAILURE;
    setup.set_cell();
    if (setup.worldlines())
        return EXIT_FAILURE;
    setup.setConstants();

    std::unique_ptr<PotentialBase> potential;
    try {
        if (bench.kind == "interaction")
            potential.reset(setup.interactionPotential());
        else
            potential.reset(setup.externalPotential());
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    if (!potential) {
        std::cerr << "error: failed to create " << bench.kind
                  << " potential '" << bench.potential << "'" << std::endl;
        return EXIT_FAILURE;
    }

    auto samples = makeSamples(bench.samples, bench.minRadius, bench.maxRadius);
    if (bench.checkBatch)
        appendGpuCheckSamples(samples);
    const auto flatSamples = flattenSamples(samples);

    std::cout << "potential=" << bench.potential
              << " kind=" << bench.kind
              << " ndim=" << NDIM
              << " samples=" << samples.size()
              << " iterations=" << bench.iterations
#ifdef USE_GPU
              << " gpu=" << (bench.gpu ? "on" : "off")
#else
              << " gpu=unavailable"
#endif
              << std::endl;

    if (bench.checkBatch) {
        if (bench.checkGpu) {
#ifdef USE_GPU
            if (!dynamic_cast<GPPotential*>(potential.get())) {
                std::cerr << "error: --benchmark-check-gpu is only supported for GPPotential" << std::endl;
                return EXIT_FAILURE;
            }
#else
            std::cerr << "error: --benchmark-check-gpu requires a GPU-enabled build" << std::endl;
            return EXIT_FAILURE;
#endif
        }
        return checkBatchedValues(*potential, samples, checkTolerance);
    }

    if (bench.method == "value" || bench.method == "all") {
#ifdef USE_GPU
        if (bench.gpu) {
            auto* gp = dynamic_cast<GPPotential*>(potential.get());
            if (!gp) {
                std::cerr << "error: --benchmark-gpu is only supported for GPPotential" << std::endl;
                return EXIT_FAILURE;
            }
            std::vector<double> values(samples.size());
            runTimed("gpuV(r)", bench.iterations, [&](std::size_t i) {
                if ((i % samples.size()) == 0)
                    gp->gpuV(flatSamples.data(), values.data(), static_cast<int>(samples.size()));
                return values[i % samples.size()];
            });
        } else
#else
        if (bench.gpu) {
            std::cerr << "error: --benchmark-gpu requires a GPU-enabled build" << std::endl;
            return EXIT_FAILURE;
        }
#endif
        runTimed("V(r)", bench.iterations, [&](std::size_t i) {
            return potential->V(samples[i % samples.size()]);
        });
    }

    if (bench.method == "gradient" || bench.method == "all") {
        runTimed("gradV(r)", bench.iterations, [&](std::size_t i) {
            const dVec grad = potential->gradV(samples[i % samples.size()]);
            return dot(grad, grad);
        });
    }

    if (bench.method == "pair" || bench.method == "all") {
        runTimed("V(r,r2)", bench.iterations, [&](std::size_t i) {
            return potential->V(samples[i % samples.size()],
                    samples[(i + 1) % samples.size()]);
        });
    }

    if (bench.method == "action" || bench.method == "all") {
        MTRand random(1234);
        Container* boxPtr = setup.get_cell();
        std::unique_ptr<PotentialBase> interaction(setup.interactionPotential());
        DynamicArray<dVec,1> initialPos =
            potential->initialConfig(boxPtr, random, constants()->initialNumParticles());
        LookupTable lookup(boxPtr, constants()->numTimeSlices(),
                constants()->initialNumParticles());
        Path path(boxPtr, lookup, constants()->numTimeSlices(),
                initialPos, constants()->numBroken());
        std::unique_ptr<WaveFunctionBase> waveFunction(setup.waveFunction(path, lookup));
        std::unique_ptr<ActionBase> action(setup.action(path, lookup, potential.get(),
                    interaction.get(), waveFunction.get()));

        runTimed("actionV(slice)", bench.iterations, [&](std::size_t i) {
            const auto value = action->potential(
                    static_cast<int>(i % constants()->numTimeSlices()));
            return value[0] + value[1];
        });
    }

    return EXIT_SUCCESS;
}
