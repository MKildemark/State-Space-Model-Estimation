# Analysis of Shock Series Data

# Install packages (uncomment and run only once if packages aren't installed)
# using Pkg
# Pkg.add(["XLSX", "DataFrames", "Plots", "StatsPlots", "Statistics", "Distributions", "HypothesisTests"])

# Load required packages
using XLSX
using DataFrames
using Plots
using StatsPlots
using Statistics
using Distributions
using HypothesisTests

"""
Read Excel data with specific handling requirements:
- Skip first empty row
- Use second row as column headers
- Include columns B to M, but exclude column C
- Handle "#I/T" values by treating them as missing
"""
function read_excel_series(file_path="shockseries.xlsx")
    # Read the Excel file
    xlsx_file = XLSX.readxlsx(file_path)
    sheet = xlsx_file[1]  # Assume data is in the first sheet
    
    # Read the entire data range including headers
    data_range = sheet[:]
    
    # Extract column headers from the second row (row index 2)
    headers = data_range[2, :]
    
    # Create a DataFrame from data, starting from row 3
    df = DataFrame(data_range[3:end, :], Symbol.(headers))
    
    # Identify columns to keep (B to M, excluding C)
    # Column A contains dates (index 1), B is index 2, C is index 3, etc.
    keep_cols = [2, 4:13...]  # Columns B, D through M
    
    # Extract only the columns we want
    df = df[:, keep_cols]
    
    # Replace "#I/T" with missing values
    for col in names(df)
        df[!, col] = replace.(df[!, col], "#I/T" => missing)
    end
    
    # Remove rows with missing values
    df = dropmissing(df)
    
    # Convert all data to Float64
    for col in names(df)
        df[!, col] = parse.(Float64, df[!, col])
    end
    
    return df
end

"""
Generate basic statistics for a data series
"""
function print_statistics(data, series_name)
    println("\n==== Basic Statistics for '$series_name' ====")
    println("Count: $(length(data))")
    println("Min: $(minimum(data))")
    println("Max: $(maximum(data))")
    println("Mean: $(mean(data))")
    println("Median: $(median(data))")
    println("Standard Deviation: $(std(data))")
end

"""
Test if a data series follows a normal distribution
"""
function test_normality(data, series_name)
    println("\n==== Normality Tests for '$series_name' ====")
    
    # Shapiro-Wilk test (better for smaller samples, n < 2000)
    sw_test = ShapiroWilkTest(data)
    
    # Kolmogorov-Smirnov test (comparing to normal distribution)
    μ = mean(data)
    σ = std(data)
    ks_test = ExactOneSampleKSTest(data, Normal(μ, σ))
    
    # Print results
    println("Shapiro-Wilk Test:")
    println("W = $(sw_test.W), p-value = $(pvalue(sw_test))")
    println("$(pvalue(sw_test) < 0.05 ? "Reject" : "Fail to reject") null hypothesis of normality at 5% significance level")
    
    println("\nKolmogorov-Smirnov Test:")
    println("D = $(ks_test.δ), p-value = $(pvalue(ks_test))")
    println("$(pvalue(ks_test) < 0.05 ? "Reject" : "Fail to reject") null hypothesis of normality at 5% significance level")
    
    # Print skewness and kurtosis as additional indicators
    skew = skewness(data)
    kurt = kurtosis(data)
    println("\nSkewness: $(round(skew, digits=3)) (0 for normal distribution)")
    println("Excess Kurtosis: $(round(kurt, digits=3)) (0 for normal distribution)")
    
    # Return test results
    return (shapiro_wilk=sw_test, kolmogorov_smirnov=ks_test, skewness=skew, kurtosis=kurt)
end

"""
Create histogram with PDF overlay for a data series
"""
function plot_histogram_with_pdf(data; title="Distribution Analysis", bins=30)
    # Calculate mean and standard deviation
    μ = mean(data)
    σ = std(data)
    
    # Create normal distribution
    dist = Normal(μ, σ)
    
    # Create histogram
    h = histogram(data, 
                 bins=bins, 
                 normalize=true,
                 alpha=0.6,
                 label="Histogram", 
                 title=title,
                 xlabel="Value", 
                 ylabel="Density")
    
    # Add estimated PDF curve
    x_range = range(minimum(data), maximum(data), length=100)
    plot!(h, x_range, pdf.(dist, x_range), 
          line=(:red, 3), 
          label="Normal PDF (μ=$(round(μ, digits=2)), σ=$(round(σ, digits=2)))")
    
    # Add kernel density estimate
    density!(h, data, label="KDE", line=(:black, 2), legend=:topright)
    
    return h
end

"""
Create a summary table of normality test results
"""
function create_normality_summary(test_results)
    summary = DataFrame(
        Series = String[],
        SW_Statistic = Float64[],
        SW_PValue = Float64[],
        KS_Statistic = Float64[],
        KS_PValue = Float64[],
        Skewness = Float64[],
        Kurtosis = Float64[],
        Is_Normal = String[]
    )
    
    for (series_name, result) in test_results
        is_normal = (pvalue(result.shapiro_wilk) > 0.05 && 
                     pvalue(result.kolmogorov_smirnov) > 0.05) ? "Yes" : "No"
        
        push!(summary, (
            series_name,
            result.shapiro_wilk.W,
            pvalue(result.shapiro_wilk),
            result.kolmogorov_smirnov.δ,
            pvalue(result.kolmogorov_smirnov),
            result.skewness,
            result.kurtosis,
            is_normal
        ))
    end
    
    return summary
end

"""
Main function to analyze all shock series
"""
function analyze_shock_series(file_path="shockseries.xlsx"; bins=25)
    # Read the data
    println("Reading data from $file_path...")
    df = read_excel_series(file_path)
    println("Successfully loaded $(size(df, 1)) rows and $(size(df, 2)) series.")
    
    # Show the first few rows
    println("\nFirst 5 rows of data:")
    println(first(df, 5))
    
    # Analyze each series
    test_results = Dict()
    individual_plots = Dict()
    
    for series_name in names(df)
        data = df[!, series_name]
        
        # Calculate and display statistics
        print_statistics(data, series_name)
        
        # Test for normality
        test_results[series_name] = test_normality(data, series_name)
        
        # Create histogram with PDF
        individual_plots[series_name] = plot_histogram_with_pdf(
            data, 
            title="Distribution of $series_name", 
            bins=bins
        )
        
        # Display individual plot
        display(individual_plots[series_name])
        
        # Save individual plot if desired
        savefig(individual_plots[series_name], "$(series_name)_distribution.png")
    end
    
    # Create combined plot
    all_plots = values(individual_plots) |> collect
    final_plot = plot(all_plots..., 
                      layout=(3, 4), 
                      size=(1200, 900), 
                      margin=5Plots.mm)
    
    # Display combined plot
    display(final_plot)
    
    # Save combined plot
    savefig(final_plot, "all_series_distributions.png")
    
    # Create and display summary table
    normality_summary = create_normality_summary(test_results)
    println("\nNormality Test Summary:")
    println(normality_summary)
    
    return df, individual_plots, final_plot, normality_summary, test_results
end

# Run the analysis
df, individual_plots, final_plot, normality_summary, test_results = analyze_shock_series("shockseries.xlsx")