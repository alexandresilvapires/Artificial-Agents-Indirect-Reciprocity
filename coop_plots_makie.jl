using CairoMakie, Colors
include("./utils.jl")

# This alternative plot code is used to present only 1 setting

const line_colors = (RGB(0.3686, 0.5059, 0.7098), RGB(0.8784, 0.6118, 0.1412), RGB(0.5608,0.6902,0.1961), RGB(0.9216, 0.3843, 0.2078), RGB(0.5, 0.5, 0.5))
const markerTypes = (:circle, :rect, :diamond, :cross, :utriangle, :xcross, :vline)
const letter_labels = ["a)", "b)", "c)", "d)", "e)", "f)","g)","h)"]
const SN_labels = ["Image Score", "Simple-Standing", "Shunning", "Stern-Judging", "All Good"]

# Set folder name here
const foldername::String = "example_coop_study"

# Obtain values from the folder
fractions = utils.parse_float32_array(String(utils.getParameterValue("Plots/$foldername/","Fractions")))
fractions = filter(x -> x <= 0.6, fractions)
results = utils.processResults("Plots/$foldername/","Disc")

function plot_single_coop_indexes(results,x_axis_scale, title, letter, folder_path,filename, keepLeg, x_axistitle="Fraction of Artificial Agents") 
    # Make plot folder
    plot_path = joinpath("Plots",folder_path)
    isdir(plot_path) || mkdir(plot_path)

    addedHeight = (title != "" ? 20 : 0)
    f = Figure(backgroundcolor = :transparent, size = (700, 400+addedHeight))

    # Define when plot parts appear.
    keepLegend = keepLeg
    XAxisTitle = x_axistitle
    YAxisTitle = "Cooperation Index, I"

    ax = Axis(f[1,1], titlesize=24, xlabelsize=24, ylabelsize=24, 
        xticklabelsize=20, yticklabelsize=20,
        title=title, xlabel=XAxisTitle, ylabel=YAxisTitle,
        xticks=0:0.1:1.01, yticks=(0:0.2:1.0),xautolimitmargin=(0.0,0.01),
        yautolimitmargin=(0.06,0.06), yminorticksvisible=true)

    for v in eachindex(results)
        # Add lines incrementally for each norm
        scatterlines!(ax, x_axis_scale, results[v][1:length(x_axis_scale)], linewidth=3, label=SN_labels[v], color=line_colors[v],marker=markerTypes[v], markersize=15)
    end

    if (keepLegend) axislegend("Social Norms",orientation = :vertical, nbanks=1, position = :rb) end

    text!(ax, 1, 1, text = letter, font = :bold, align = (:left, :top), offset = (-585, -2),
        space = :relative, fontsize = 24
    )

    # Save the plot inside the folder
    final_path = joinpath(plot_path, filename*"_coop_index.pdf")
    save(final_path, f)
end

plot_single_coop_indexes(results, fractions, "", "",foldername,"final", true)