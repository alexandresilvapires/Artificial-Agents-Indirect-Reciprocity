include("./utils.jl")
using Plots, Measures
using LaTeXStrings

# Set norms and AA strategy used
sn_tags = ["IS","SS","SH","SJ"]
aa_strat = "Disc"

axisTitles = ["Execution Error, ε", "Assignment Error, α", "Assessment Error, χ"]

# Set line titles 
lineTitles = ["A = 0","A = 8 (FSAs)","A = 8 (G FRSAs)"]

# Set folder path
folder_title = "example_error_study"

fractions = utils.parse_float32_array(String(utils.getParameterValue("Plots/"*folder_title,"Error Fractions")))

plot_font = "Arial"
default(fontfamily=plot_font)

function get_res_with_tag(tags::Vector,errorName::String)
    return [utils.processResults("Plots/"*folder_title,tags[1]*errorName), utils.processResults("Plots/"*folder_title,tags[2]*errorName), utils.processResults("Plots/"*folder_title,tags[3]*errorName)]
end

function make_plot_folder(folder_name)::String
    # Create the path for the folder
    folder_path = joinpath("Plots", folder_name)

    # Create the folder if it doesn't exist
    isdir(folder_path) || mkdir(folder_path)

    return folder_path
end

folder_path = make_plot_folder(folder_title)

# Takes 3 result arrays edited from a results.txt and plots them side by side using a common label and a more readable format
function plot_three_coop_indexes(results, sn_tag, fractions_error, axisTitles, lineTitles, folder_path, line_color_default) 

    line_colors = (line_color_default, line_color_default.*0.7, line_color_default.*0.1)

    subplots = []

    l = @layout [a b c]

    for i in eachindex(results)
        # Create a single plot for each result

        keepLegend = (i == length(results) ? :left : false)
        XAxisTitle = (i == Int(round(length(results)/2)) ? axisTitles[i] : axisTitles[i])
        YAxisTitle = (i == 1 ? L"Cooperation Index, $\mathit{I}$" : "")

        tick_positions = [1e-4, 1e-3, 1e-2, 0.1, 0.5]
        tick_labels = ["1e-4","1e-3","1e-2", 0.1, 0.5]
        p = plot(xlabel=XAxisTitle, xscale=:log10, xticks=(tick_positions, tick_labels), 
        yticks=(0:0.2:1, 0:0.2:1), 
        xlims=(minimum(tick_positions), maximum(tick_positions)+0.02), ylims=(0, 1),
        frame=:true, grid=:true, 
        ylabel=YAxisTitle, 
        size=(1600, 340), dpi=1000, bottom_margin=0mm, tickfont=font(10), 
        xguidefontsize=20, yguidefontsize=20)

        if i > 1 p = plot!(p,yformatter=_->"", left_margin=0mm) end
        
        res = results[i]

        for v in eachindex(res)
            # Add lines incrementally for each norm
            plot!(p, fractions_error, res[v], label=lineTitles[v], linecolor=line_colors[v],linewidth=2,legend=keepLegend,marker=(:circle,3),markercolor=line_colors[v],markerstrokewidth=0)
        end

        push!(subplots, p)
    end
    totalplot = plot(subplots[1], subplots[2], subplots[3], layout=l,legendfontsize=10, bottom_margin=12mm, right_margin=0mm)
    totalplot = plot!(totalplot, left_margin = [10mm 0mm 0mm])

    # Save the plot inside the folder
    plot_path = joinpath(folder_path, "error"*sn_tag*"_plot.png")
    savefig(totalplot, plot_path)
end

for sn_index in eachindex(sn_tags)
    sn_tag = sn_tags[sn_index]
    tags = [aa_strat*":"*sn_tag*"_NoAAs",aa_strat*":"*sn_tag*"_AAs",aa_strat*":"*sn_tag*"_FRAAs"]

    results = [get_res_with_tag(tags, "ExecError"), get_res_with_tag(tags, "AssignError"), get_res_with_tag(tags, "AssessError")]

    colors = RGB(0.3686, 0.5059, 0.7098), RGB(0.8784, 0.6118, 0.1412), RGB(0.5608,0.6902,0.1961), RGB(0.9216, 0.3843, 0.2078)

    plot_three_coop_indexes(results, sn_tag, fractions, axisTitles, lineTitles, folder_path, colors[sn_index]) 
end