#include "common.h"
#include "container.h"
#include "factory_potential.h"
#include "potential.h"
#include "setup.h"

#include <chrono>
#include <cstdlib>
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
        << "  --benchmark-method METHOD        value, gradient, pair, or all\n"
        << "  --benchmark-iterations N         calls per measured method [1000000]\n"
        << "  --benchmark-samples N            generated sample positions [4096]\n"
        << "  --benchmark-min R                minimum sample radius [0.25]\n"
        << "  --benchmark-max R                maximum sample radius [5.0]\n\n"
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

} // namespace

int main(int argc, char* argv[])
{
    BenchmarkOptions bench;
    std::vector<std::string> setupArgs;
    setupArgs.push_back(argv[0]);

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);

            if (arg == "--benchmark-help") {
                bench.help = true;
            } else if (arg == "--benchmark-list") {
                bench.list = true;
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
            bench.method != "pair" && bench.method != "all") {
        std::cerr << "error: --benchmark-method must be value, gradient, pair, or all" << std::endl;
        return EXIT_FAILURE;
    }
    if (bench.iterations == 0 || bench.samples == 0 ||
            bench.minRadius <= 0.0 || bench.maxRadius <= bench.minRadius) {
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

    const auto samples = makeSamples(bench.samples, bench.minRadius, bench.maxRadius);

    std::cout << "potential=" << bench.potential
              << " kind=" << bench.kind
              << " ndim=" << NDIM
              << " samples=" << samples.size()
              << " iterations=" << bench.iterations
              << std::endl;

    if (bench.method == "value" || bench.method == "all") {
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

    return EXIT_SUCCESS;
}
