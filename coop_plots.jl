include("./utils.jl")
using Plots, Measures
using LaTeXStrings

# Up to 3 result files can be used simultaneously

# Default way for 3 plots
# Set folder paths here
foldername_1 = "Plots/example_coop_study/" 
foldername_2 = ""
foldername_3 = ""

disc_baseline = utils.processResults(foldername_1,"Disc")
#disc_fixed_g = utils.processResults(foldername_2,"Disc")
#disc_fixed_b = utils.processResults(foldername_3,"Disc")

# Put all used results here
results = [disc_baseline]

# Set titles for each plot, and folder where to store final results
titles = ["FSA","G FRSAs","B FRSAs"]
folder_title = "example_coop_plots_combined"

fractions = utils.parse_float32_array(String(utils.getParameterValue(foldername_1,"Fractions")))
fractions = filter(x -> x <= 0.6, fractions)    # Only show up to 0.6 AAs

plot_font = "Arial"
default(fontfamily=plot_font)

function make_plot_folder(folder_name)::String
    # Create the path for the folder
    folder_path = joinpath("Plots", folder_name)

    # Create the folder if it doesn't exist
    isdir(folder_path) || mkdir(folder_path)

    return folder_path
end


# Takes 3 result arrays edited from a results.txt and plots them side by side using a common label and a more readable format
function plot_three_coop_indexes(results,fractions_AAs, subplot_titles, folder_name) 
    folder_path = make_plot_folder(folder_name)

    line_colors = (RGB(0.3686, 0.5059, 0.7098), RGB(0.8784, 0.6118, 0.1412), RGB(0.5608,0.6902,0.1961), RGB(0.9216, 0.3843, 0.2078), RGB(0.5, 0.5, 0.5))
    markerTypes = [:circle, :square, :dtriangle, :utriangle, :diamond]
    SN_labels = ["Image Score", "Simple-Standing", "Shunning", "Stern-Judging", "All Good"]

    subplots = []

    l = @layout [a b c]
    if length(results) == 2 
        l = @layout [a b]
    end

    for i in eachindex(results)
        # Create a single plot for each result
        keepLegend = (i == length(results) ? :bottomright : false)
        XAxisTitle = "Fraction of Artificial Agents"
        YAxisTitle = (i == 1 ? L"Cooperation Index, $\mathit{I}$" : "")

        p = plot(xlabel=XAxisTitle, xticks=0:0.1:maximum(fractions_AAs)+0.04, yticks=(0:0.2:1, 0:0.2:1), xlims=(0, maximum(fractions_AAs)+0.02), ylims=(0, 1),
            frame=:true, grid=:true,
            ylabel=YAxisTitle, title=subplot_titles[i],
            size=(1600*length(results)/2.5, 340), dpi=1000, bottom_margin=0mm, tickfont=font(12),xguidefontsize=20, yguidefontsize=20,legendfontsize=11)
        
        if i > 1 p = plot!(p,yformatter=_->"", left_margin=0mm) end
        
        res = results[i]

        for v in eachindex(res)
            # Add lines incrementally for each norm
            plot!(p, fractions_AAs, res[v][1:length(fractions_AAs)], label=SN_labels[v], linecolor=line_colors[v],linewidth=2,legend=keepLegend,marker=(markerTypes[v],4),markercolor=line_colors[v],markerstrokewidth=0)
        end

        push!(subplots, p)
    end
    if length(results) == 3
        totalplot = plot(subplots[1], subplots[2], subplots[3], layout=l, bottom_margin=14mm, right_margin=0mm)
    elseif length(results) == 2
        totalplot = plot(subplots[1], subplots[2], layout=l, bottom_margin=12mm, right_margin=0mm)
    else
        totalplot = plot(subplots[1], layout=l, bottom_margin=7mm, right_margin=0mm)
    end
    totalplot = plot!(totalplot, left_margin = [12mm 0mm 0mm])
    # Save the plot inside the folder
    plot_path = joinpath(folder_path, "resulting_plot_noimitnew.png")
    savefig(totalplot, plot_path)
end

plot_three_coop_indexes(results, fractions, titles, folder_title) 
