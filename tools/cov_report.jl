# Parse *.cov files in the CUDA ext dir and report, per top-level definition,
# the max execution count across its body lines. A host function with max==0
# never ran. Device kernels (name contains "kernel") run on the GPU where line
# coverage is blind, so their 0 is not evidence of deadness — flagged [kernel].
const COVDIR = raw"C:\Users\loisel\Dropbox\2026\nonuniform\MultiGridBarrier.jl\ext\MultiGridBarrierCUDAExt"

defstart(src) = occursin(r"^(function |const |[A-Za-z_][\w.!:{}, <:]*\(.*\) where|[A-Za-z_][\w.!]*\(.*\)\s*=|MultiGridBarrier\.|Base\.|_\w+\()", src)

for f in filter(x->endswith(x,".cov"), readdir(COVDIR; join=true))
    base = basename(f)
    rows = Tuple{Int,String,Union{Int,Nothing}}[]  # (srcline, src, count|nothing)
    for (i,ln) in enumerate(eachline(f))
        cnt = nothing
        m = match(r"^\s*(\d+) (.*)$", ln)
        if m !== nothing
            cnt = parse(Int, m.captures[1]); src = m.captures[2]
        else
            mm = match(r"^\s*- (.*)$", ln); src = mm===nothing ? ln : mm.captures[1]
        end
        push!(rows, (i, src, cnt))
    end
    # Walk: a def line opens a block; accumulate counts until the next def line
    # at the same (zero) indentation or EOF.
    println("\n==== $base ====")
    n = length(rows)
    i = 1
    while i <= n
        (ln, src, _) = rows[i]
        stripped = lstrip(src)
        iszerocol = startswith(src, stripped)  # def must be at column 0
        if iszerocol && defstart(src) && !startswith(stripped, "#") && !startswith(stripped,"import") && !startswith(stripped,"using") && !startswith(stripped,"export")
            j = i+1; mx = nothing; lastbody = i
            while j <= n
                (lj, sj, cj) = rows[j]
                sjl = lstrip(sj)
                if startswith(sj, sjl) && defstart(sj) && !startswith(sjl,"#") && length(sjl)>0
                    break
                end
                if cj !== nothing; mx = mx===nothing ? cj : max(mx,cj); end
                if length(sjl)>0; lastbody = j; end
                j += 1
            end
            iskern = occursin("kernel", src)
            tag = iskern ? "[kernel]" : (mx===nothing ? "[no-exec-lines]" : (mx==0 ? "*** NEVER RAN ***" : "ran(max=$mx)"))
            println(rpad("L$ln", 7), " ", rpad(tag,20), " ", first(src, min(length(src),90)))
            i = j
        else
            i += 1
        end
    end
end
println("\nCOV_REPORT_DONE")
