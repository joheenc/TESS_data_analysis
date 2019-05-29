include("/rigel/home/jc5110/sebastien.jl")
using Main.WeirdDetector
using PyPlot
using DataFrames
using CSV

stars = DataFrame(TID=Int64[])
open("/rigel/home/jc5110/TESSsector1/tesscurl_sector_1_lc.sh") do file
    c = Int64
    c = 1
    for ln in eachline(file)
        tid = Int64
	if c >= parse(Int64, ARGS[1]) && c < parse(Int64, ARGS[2])
            try tid = parse(Int64, ln[length(ln)-30:length(ln)-15]) catch; end
            push!(stars, tid)
	end
	c = c+1
    end
end

out = DataFrame(TID=Int64[], Peak_Zeta=Float64[], Guessed_Period=Float64[])
for i in stars[1]
    tid = string(i)
    df = loadFITS(0, fitsdir="/rigel/astro/users/jc5110/TESSsector1/", tic_id=tid)
    d = pointsify(df)
    periods = optimal_periods(0.25,15)
    output=periodogram(d, periods, parallel=true, datakw=true)
    output[:delt_chi2] = flatten(periods, output[:chi2], tess=true)
    null_output = scrambled_periodogram(df, periods, tess=true)
    null_output[:delt_chi2] = flatten(periods, null_output[:chi2], tess=true)
    sigma = movingstd((null_output[
    :kurtosis] .- 3) .* (null_output[:delt_chi2]))
    output[:zeta] = (output[:kurtosis] .- 3) .* output[:delt_chi2] ./ sigma

    plot(periods, output[:zeta])
    savefig("/rigel/home/jc5110/TESSsector1/periodograms/" * tid * ".png")
    clf()
    period = periods[findall(a->a==maximum(output[:zeta]), output[:zeta])][1]
    df[:t] = df[:t] .% period
    scatter(df[:t], df[:F], alpha=0.4, s=0.1)
    savefig("/rigel/home/jc5110/TESSsector1/foldedlightcurves/" * tid * ".png")
    clf()
    zetapeak = maximum(output[:zeta])
    push!(out, [i, zetapeak, period])
    CSV.write("/rigel/home/jc5110/TESSsector1/sector1data-" * ARGS[1] * "-" * ARGS[2] * ".csv", out)
end
close(f)
