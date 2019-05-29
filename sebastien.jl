module WeirdDetector

    using Interpolations
    using DataFrames
    using FITSIO
    using Optim
    using Statistics
    using Distributed
    using Random

    export Point, periodogram, aliases, optimal_periods, getFITS, loadFITS, pointsify, flatten, scrambled_periodogram, movingstd

    struct Point
        t :: Float32
        modt :: Float32
        F :: Float32
        sigmaF :: Float32
        smoothedF :: Float32
    end
    Point(t::Real, F::Real, sigmaF::Real) = Point(Float32(t), NaN, Float32(F), Float32(sigmaF), NaN)
    Point(t::Real, F::Real) = Point(t, F, 5f-5)

    """
    convert a DataFrame in the format returned by `loadFITS`
    to an array of `Points`
    """
    function pointsify(df; keep_interpolated::Bool=false) :: Vector{Point}
        if keep_interpolated
            df[df[:interpolated] .== true, :sigmaF] = mean(df[df[:interpolated] .== false, :sigmaF])
        else
           df = df[df[:interpolated] .== false, :]
        end
        Point.((df[:t]), (df[:F]), (df[:sigmaF]))
    end

    function fold!(period::Float32, data::Array{Point})
        for (i,p) in enumerate(data)
            data[i] = Point(p.t, p.t%period, p.F, p.sigmaF, p.smoothedF)
        end
        sort!(data, by=(x -> x.modt))
    end

    "helper function for smoothing"
    function access(data::Vector{A}, i::Int) :: A where A
        npoints = length(data)
        j = mod(i,npoints) == 0 ? npoints : mod(i,npoints)
        data[j]
    end

    "helper function for smoothing"
    function wrappedtime(data::Vector{Point}, i::Int, P::Float32) :: Float32
        npoints = length(data)
        modt = access(data, i).modt
        if i < 1
            return modt - P
        elseif i > npoints
            return modt + P
        else
            return modt
        end
    end



    function smooth2!(data::Vector{Point}, P::Float32, kw::Float32)
        kw = Int(round(kw * length(data)))
        s = map(collect(-kw:kw)) do i
            p = access(data, i)
            (p.F / p.sigmaF, 1/p.sigmaF)
        end
        sumFoverSigma = sum(first.(s))
        sumInverseSigma = sum(last.(s))
        lb = -kw
        ub = kw
        for i = 1:length(data)
            point = access(data, lb)
            sumFoverSigma -= point.F / point.sigmaF
            sumInverseSigma -= 1e0 / point.sigmaF
            lb += 1

            ub += 1
            point = access(data, ub)
            sumFoverSigma += point.F / point.sigmaF
            sumInverseSigma += 1e0 / point.sigmaF

            data[i] = Point(data[i].t, data[i].modt, data[i].F, data[i].sigmaF,
                            Float32(sumFoverSigma/sumInverseSigma))
        end
    end

    "smooth the data with a rolling mean"
    function smooth!(data::Vector{Point}, P::Float32, kw::Float32)
        kw = kw * P
        sumFoverSigma = 0e0
        sumInverseSigma = 0e0
        lb = 2
        ub = 1
        while wrappedtime(data, lb, P) > data[1].modt - kw
            lb -= 1
            point = access(data, lb)
            sumFoverSigma += point.F / point.sigmaF
            sumInverseSigma += 1e0 / point.sigmaF
        end
        for i = 1:length(data)
            while wrappedtime(data, lb, P) < data[i].modt - kw
                point = access(data, lb)
                sumFoverSigma -= point.F / point.sigmaF
                sumInverseSigma -= 1e0 / point.sigmaF
                lb += 1
            end
            while wrappedtime(data, ub, P) < data[i].modt + kw
                ub += 1
                point = access(data, ub)
                sumFoverSigma += point.F / point.sigmaF
                sumInverseSigma += 1e0 / point.sigmaF
            end
            data[i] = Point(data[i].t, data[i].modt, data[i].F, data[i].sigmaF,
                            Float32(sumFoverSigma/sumInverseSigma))
        end
    end

    "calculate reduced chi squared on data that has already been folded and smoothed"
    function chi2(data::Vector{Point}) :: Float32
        chi2 = 0e0
        for p in data
            chi2 += ((p.F - p.smoothedF)/p.sigmaF)^2
        end
        Float32(chi2)
    end

    "fold, smooth, and calculate reduced chi squared"
    function chi2(data::Vector{Point}, period::Float32; kw::Float32=0.002f0)
        fold!(period, data)
        smooth!(data, period, kw)
        chi2(data)
    end

    function kurtosis(data::Vector{Point}) :: Float32
        fs = (x->x.smoothedF).(data)
        mu = mean(fs)
        sigma = std(fs)
        mean((fs .- mu).^4)/sigma^4
    end

    "returns dataframe containing chi-squared and kurtosis by period"
    function periodogram(data::Vector{Point}, periods::Vector{Float32}; kw=0.002f0,
                         parallel=true, datakw=false, postprocess=true)
        df = DataFrame(chi2=Float32[], kurtosis=Float32[])
        s = div(length(periods), nworkers()*3)
        results = pmap(periods, distributed=parallel, batch_size=s) do p
            fold!(p, data)
            row = Vector{Any}()
            if datakw
                smooth2!(data, p, kw)
            else
                smooth!(data, p, kw)
            end
            push!(row, chi2(data))
            push!(row, kurtosis(data))
            push!(df, row)
        end
        df
    end

    function aliases(downto::Int=5; upperBound=nothing) :: Vector{Rational}
        lines = Set{Rational}()
        for n in 1:downto
            for m in 1:n
                if upperBound == nothing
                    push!(lines, m//n)
                else
                    i = 0
                    while i + m//n < upperBound
                        push!(lines, i + m//n)
                        i += 1
                    end
                end
            end
        end
        collect(lines)
    end

    function flatten(periods::Vector{Float32}, chi2s::Vector{Float32}; stepwidth::Float32=10.21f0,
                     preflipped=false, tess=false)
        nchi2s = similar(chi2s)
        if (!tess)
            steps = Int(ceil(periods[end]/stepwidth))
            for n in 1:steps
                lb = first(searchsorted(periods, (n-1)*stepwidth))
                ub = last(searchsorted(periods, n*stepwidth))
                line(x, p) = p[1]*x + p[2]
                c2(p) = sum((line(periods[lb:ub], p) .- chi2s[lb:ub]).^2)
                p0 = [0e0, 0e0]
                p0[1] = (chi2s[ub] - chi2s[lb])/(periods[ub] - periods[lb])
                p0[2] = chi2s[lb] - p0[1]*periods[lb]
                res = Optim.optimize(c2, p0, Newton())
                p = Optim.minimizer(res)
                nchi2s[lb:ub] .= Float32.(line(periods[lb:ub], p) - chi2s[lb:ub])
            end
        else
            medianchi2s = fill(median(chi2s), length(chi2s))
            nchi2s = medianchi2s - chi2s
        end
        if preflipped
            -nchi2s
        else
            nchi2s
        end
    end

    function movingstd(npowers::Vector{Float32}, kw=200) :: Vector{Float32}
        sigmas = similar(npowers)
        for i in 1:length(npowers)
            lb = i - kw < 1 ? 1 : i - kw
            ub = i + kw > length(npowers) ? length(npowers) : i + kw
            sigmas[i] = std(npowers[lb:ub])
        end
        sigmas
    end

    function scrambled_periodogram(df::DataFrame, periods::Vector{Float32}; tess=false, kwargs...)
        df = deepcopy(df)
        dropmissing!(df, :F, disallowmissing=false)
        df[:F] = df[randperm(size(df)[1]), :F]
        df[:sigmaF] = df[randperm(size(df)[1]), :sigmaF]
        data = pointsify(df)
        periodogram(data, periods; kwargs...)
    end

    function optimal_periods(pmin=0.25f0, pmax=5f1; n=6)
        pmin = Float32(pmin)
        pmax = Float32(pmax)
       exp.((log(pmin) : (0.001f0/n) : log(pmax)))
    end

    "download FITS data for KIC"
    getFITS(KIC::Int; fitsdir="fitsfiles/") = getFITS(lpad(KIC,9,0); fitsdir=fitsdir)
    function getFITS(KIC::String; fitsdir="fitsfiles/", force=false, tic_id="", ete6=false)
        if tic_id == ""
            #info("Dowloading data for KIC $(KIC)")
            ftpfolder = "http://archive.stsci.edu/pub/kepler/lightcurves//" * KIC[1:4] * "/" * KIC * "/"
        elseif ete6
            tic_id = lpad(tic_id, 9, 0)
            ftpfolder = "http://archive.stsci.edu/missions/tess/ete-6/tid/00/000/00" * tic_id[1:1] * "/" * tic_id[2:4] * "/tess2019128220341-" * lpad(tic_id,16,0) * "-0016-s_lc.fits"
        else
            tic_id = lpad(tic_id, 9, 0)     #good for sector 1 data only
            sector = lpad(sector, 2, 0)
            ftpfolder = "https://mast.stsci.edu/api/v0.1/Download/file/?uri=mast:TESS/product/tess2018206045859-s0001-" * lpad(tic_id,16,0) * "-0120-s_lc.fits"
        end
        command = `wget -P $fitsdir -nH --cut-dirs=6 -r -c -N -np -q -R '*.tar' -R 'index*' -erobots=off $ftpfolder`
        run(command)
    end

    #window width is kw*2+1
    "remove outliers from time series.  This is a hleper function for loadFITS"
    function prune(df::DataFrame, threshold_sigma::Int, kw::Int; usephoto=false) :: DataFrame
        nrows = size(df,1)
        med = map(1:nrows) do i
            lb = i-kw < 1 ? 1 : i-kw
            ub = i+kw < nrows ? i+kw : nrows
            m = median(df[:F][lb:ub])
            m, median(abs.(m .- df[:F][lb:ub]))
        end
        df = df[1+kw : end-kw, :]
        med = med[1+kw : end-kw]
        sigma = (m->m[2]).(med)
        med = (m->m[1]).(med)
        if usephoto
            df[abs.(df[:F] .- med) .< threshold_sigma.*df[:sigmaF], :]
        else
            df[abs.(df[:F] .- med) .< threshold_sigma.*sigma.*1.4826, :]
        end
    end

    "detrend with moving median.  This is a helper function for loadFITS"
    function detrend(df::DataFrame, P, trim=true, func=median)
        m = func(df[:F])
        df[:F] .-= m
        fs = 60*24 / 29.4 #sampling frequency [days^-1]
        kw = Int(round(P*fs/2))
        nrows = size(df, 1)
        smoothedF = Vector{Float32}(undef, nrows)
        for i in 1:nrows
            lb = i-kw < 1 ? 1 : i-kw
            ub = i+kw < nrows ? i+kw : nrows
            smoothedF[i] = func(df[:F][lb:ub])
        end
        df[:F] .= df[:F] .+ m .- smoothedF
        if trim
            df[1+kw: end-kw, :]
        else
            df
        end
    end

    "Interpolate missing points from LC.  This is a helper function for loadFITS"
    function interpolate_missing!(df::DataFrame; tess::Bool=false)
        #collect points to interpolate
        df[:interpolated] = false
        newTimes = Vector{Float64}()
        dt = 0.020416 #kepler cadence in days
        if (tess)
            dt = 0.001388889 #tess cadence in days
        end
        epsilon = 0.005
        for i in 2:size(df,1)
            t = df[i-1, :t]
            while df[i,:t] - t > dt + epsilon
                t += dt
                push!(newTimes, t)
            end
        end
        itp = Interpolations.interpolate((Vector{Float32}(df[:t]),),
                                         Vector{Float32}(df[:F]), Gridded(Linear()))
        for t in newTimes
            if (!tess)
                row = [t, itp(t), missing, true]
            else
                row = [t, itp(t), -1, true]
            end
            push!(df, row)
        end
        #sort in interpolated points
        sort!(df, [:t])
    end

    """
    returns a DataFrame containing the lightcurve across all quarters for KIC
    with outliers removed and missing and bad chi2s = chi2(data, periods) points interpolated.
    """
    loadFITS(KIC::Int; kwargs...) = loadFITS(lpad(string(KIC),9,"0"); kwargs...)
    function loadFITS(KIC::String; usetimecorr=true, prune_kw=5, prune_theshold=5,
                      prune_usephoto=false, usephasmaP=0, cadence="llc",
                      nodetrend=false, detrend_kw=2, splitwidth=0.3, trim=true, detrend_with=median,
                      quarters=1:16, fitsdir="fitsfiles/", usePDC=true, tic_id="", ete6=false) :: DataFrame

	filenames = []

        if (tic_id=="")
            filenames = [fitsdir*fn for fn in readdir(fitsdir)
                         if occursin(KIC, fn) && occursin(cadence, fn)]
        elseif ete6
            tic_id = lpad(tic_id, 16, "0")
            fitsdir = fitsdir * "/" * tic_id[6:8] * "/" * tic_id[9:11] * "/"
            filenames = [fitsdir*fn for fn in readdir(fitsdir)
                        if occursin(tic_id, fn) && occursin("-s_lc", fn)]
        else
            tic_id = lpad(tic_id, 16, "0")
	    for fn in readdir(fitsdir)
		if occursin(tic_id, fn) && occursin("-s_lc", fn)
			filenames = [fitsdir*fn]
		end
	    end
        end
        if filenames == []
            return DataFrame()
        end
        dfs = []

        for fn in filenames
            f = FITS(fn)
            quarter = 0
            if tic_id == ""
                quality = "SAP_QUALITY"
                quarter = read_key(f[1], "QUARTER")[1]
            else
                quality = "QUALITY"
            end

            if (tic_id != "") || (quarter in quarters)
                df = DataFrame()
                fluxerrcol = "PDCSAP_FLUX_ERR"
                if usePDC
                    fluxcol = "PDCSAP_FLUX"
                else
                    fluxcol = "SAP_FLUX"
                    if (tic_id != "")
                        fluxerrcol = "SAP_FLUX_ERR"
                    end
                end
                names = [("TIME", :t), ("TIMECORR", :tcorr), (fluxcol , :F),
                        (fluxerrcol, :sigmaF), (quality, :QUALITY)]
                for (oldname, newname) in names
                     df[newname] = [isnan(x) ? missing : x for x in read(f[2], oldname)]
                end

                if usetimecorr
                    df[:t] .+= df[:tcorr]
                end
                deletecols!(df, :tcorr)
                dropmissing!(df, :F, disallowmissing=false)

                #remove outliers
                df = prune(df, 5, 5, usephoto=false)
                #drop bad points
                df = df[df[:QUALITY] .== 0,:]
                deletecols!(df, :QUALITY)


                ts = df[:t]
                delts = [ts[i+1] - ts[i] for i in 1:(length(ts)-1)]
                boundaries = vcat([0], findall((dt->dt>splitwidth), delts), [length(ts)])
                if boundaries == [0,0]
                    continue
                end
                for i in 1:(length(boundaries)-1)
                    sdf = df[boundaries[i]+1:boundaries[i+1], :]
                    if sdf[end, :t] - sdf[1, :t] < splitwidth
                        continue
                    end
                    push!(dfs, sdf)
                end
            end
            close(f)
        end

        #detrend and normalize each segment
        for i in 1:length(dfs)
            dfs[i][:F] = convert(Vector{Float32}, dfs[i][:F])

            if (tic_id == "")
                interpolate_missing!(dfs[i])
            else
                interpolate_missing!(dfs[i], tess = true)
            end
            if usephasmaP != 0
                dfs[i] = phasma(dfs[i], usephasmaP)
            elseif !nodetrend
                dfs[i] = detrend(dfs[i], detrend_kw, trim, detrend_with)
                println(size(detrend(dfs[i], detrend_kw, trim, detrend_with)))
            end
            #normalized each quarter individually
            m = mean(dfs[i][:F])
            dfs[i][:F] ./= m
            dfs[i][:sigmaF] ./= m
            dfs[i][:F] .-= 1
        end

        if length(dfs) == 0
            return DataFrame()
        elseif length(dfs) == 1
            return dfs[1]
        else
            #combine quarters
            return vcat(dfs...)
        end
    end
end
