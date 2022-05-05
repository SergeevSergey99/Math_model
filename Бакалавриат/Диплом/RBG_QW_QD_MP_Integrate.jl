using Plots, Random, Roots, FFTW, LaTeXStrings, DynamicalSystems, StatsBase

import PyPlot;
const plt = PyPlot;

cd(dirname(@__FILE__));

# parameters
# physical cavity photon lifetime (and norm time)
tphys = 7.0 * 10^(-12);

# QW parameters
tg = 0.1 * 10^(-9);

# QD parameters
tρ = 0.1 * 10^(-9);
tn = 0.1 * 10^(-9);

delayphys = 0.7 * 10^(-9);

# normalized material parameters
# QW
etag = tphys / tg;

# QD
g = 1.15;
B = 924;
etaρ = tphys / tρ;
etan = tphys / tn;

α = 2;

# pump
JthrQW = 1;
JthrQD = ((1 + g) * (-B + g + B * g)) / (B * (-1 + g) * g);
nJthr = 1.5;
JQW = JthrQW * nJthr;
JQD = JthrQD * nJthr;

# optical feedback
τdelay = delayphys / tphys;
γ = 0.00;
ϕ = 0;
ϕs = [0, 0, 0];

# neighbor coupling
κ = 0.12;

# vectors of parameters
p0QW = [JQW, etag, α, γ, ϕ, ϕs[1], ϕs[2], ϕs[2], κ, τdelay];
p0QD = [JQD, g, B, etaρ, etan, α, γ, ϕ, ϕs[1], ϕs[2], ϕs[2], κ, τdelay];
p0QD4 = [JQD, g, B, etaρ, etan, α, γ, ϕ, ϕs[1], ϕs[2], ϕs[2], ϕs[2], κ, τdelay];

t0 = 0;

# integration procedure parameters
dt = τdelay / 10000;
hist_len = floor(Int, τdelay / dt);
endtime = 400 * τdelay;
num_iter = floor(Int, endtime / dt);
n_lasers = 3
dimQW = n_lasers * 3; # problem dimension QW
dimQD = n_lasers * 4; # problem dimension QD
savnum = 500; # number of points per delay to save
modstep = floor(Int, hist_len / savnum);

#Reind = zeros(Int,n_lasers)
#Imind = zeros(Int,n_lasers)

# switch QW and QD
isQW = false;
if isQW
    p0 = copy(p0QW)
    dim = dimQW
    tauind = dim + 1
    Reind = [1,4,7]
    Imind = [2,5,8]
    """for i = 1:n_lasers
        Reind[i] = (i - 1)*3 + 1
        Imind[i] = (i - 1)*3 + 2
    end"""
else
    p0 = copy(p0QD)
    dim = dimQD
    tauind = dim + 1
    Reind = [1,5,9]
    Imind = [2,6,10]
    """
    for i = 1:n_lasers
        Reind[i] = (i - 1)*4 + 1
        Imind[i] = (i - 1)*4 + 2
    end"""
end



## function evaluating the derivatives for the iteration with the noise terms
# QW problem
# u is the dependent variables vector for three lasers
# indices u[1:4] are for ℜ, ℑ, ρ, n for the first laser
# u[5:8] for the second
# u[9:12] for the third
# uT is the corresponding delayed dependent variables vector
# p is the parameter vector
# p[1] - J
# p[2] - etag
# p[3] - α
# p[4] - γ
# p[5] - ϕ
# p[6] - ϕs[1]
# p[7] - ϕs[2]
# p[8] - ϕs[2]
# p[9] - κ
# p[10] - τdelay

function QW(u, E1, E2, ET1, ET2, ET3, p, nt)
    dF =
        0.5 * (1 - im * p[3]) * (u[3] - 1) * (u[1] + im * u[2]) +
        im *
        p[4] *
        exp(-im * p[5]) *
        (exp(-im * p[6]) * ET1 + exp(-im * p[7]) * ET2 + exp(-im * p[8]) * ET3) +
        im * p[9] * (E1 + E2)

    dn = p[2] * (p[1] - u[3] * (1 + u[1]^2 + u[2]^2))

    return [real(dF), imag(dF), dn]
end

function CoupledQW(u, uT, p, nt, t, t0)
    du1 = QW(
        u[1:3],
        u[4] + im * u[5],
        u[7] + im * u[8],
        uT[1] + im * uT[2],
        uT[4] + im * uT[5],
        uT[7] + im * uT[8],
        p,
        nt,
    )

    du2 = QW(
        u[4:6],
        u[1] + im * u[2],
        u[7] + im * u[8],
        uT[1] + im * uT[2],
        uT[4] + im * uT[5],
        uT[7] + im * uT[8],
        p,
        nt,
    )

    du3 = QW(
        u[7:9],
        u[1] + im * u[2],
        u[4] + im * u[5],
        uT[1] + im * uT[2],
        uT[4] + im * uT[5],
        uT[7] + im * uT[8],
        p,
        nt,
    )

    return collect(Iterators.flatten([du1, du2, du3]))
end

# QD problem
# u is the dependent variables vector for three lasers
# indices u[1:4] are for ℜ, ℑ, ρ, n for the first laser
# u[5:8] for the second
# u[9:12] for the third
# uT is the corresponding delayed dependent variables vector
# p is the parameter vector
# p[1] - J
# p[2] - g
# p[3] - B
# p[4] - etaρ
# p[5] - etan
# p[6] - α
# p[7] - γ
# p[8] - ϕ
# p[9] - ϕs[1]
# p[10] - ϕs[2]
# p[11] - ϕs[2]
# p[12] - κ
# p[13] - τdelay

function FU(ρ, n, B)
    return B * n * (1 - ρ)
end

function QD(u, E1, E2, ET1, ET2, ET3, p, nt)
    dF =
        0.5 *
        ((1 - im * p[6]) * (p[2] * (2 * u[3] - 1) - 1)) *
        (u[1] + im * u[2]) +
        im *
        p[7] *
        exp(-im * p[8]) *
        (
            exp(-im * p[9]) * ET1 +
            exp(-im * p[10]) * ET2 +
            exp(-im * p[11]) * ET3
        ) +
        im * p[12] * (E1 + E2)

    dρ =
        p[4] *
        (FU(u[3], u[4], p[3]) - u[3] - (2 * u[3] - 1) * (u[1]^2 + u[2]^2))

    dn = p[5] * (p[1] - u[4] - 2 * FU(u[3], u[4], p[3]))

    return [real(dF), imag(dF), dρ, dn]
end
du = [[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]

#
function CoupledQD(u, uT, p, nt, t, t0)



    """
    du = [ [0.0,0.0,0.0,0.0] for i = 1:n_lasers  ]

    uti1 = [5,1,1]
    uti2 = [9,9,5]


    for i = 1:n_lasers
        du[i] = QD(
            u[((i - 1)*4 + 1): ((i - 1)*4 + 4)],
            u[uti1[i]] + im * u[uti1[i] + 1],
            u[uti2[i]] + im * u[uti2[i] + 1],
            uT[1] + im * uT[2],
            uT[5] + im * uT[6],
            uT[9] + im * uT[10],
            p,
            nt,
        )
    end
    """
    du[1] = QD(
        u[1:4],
        u[5] + im * u[6],
        u[9] + im * u[10],
        uT[1] + im * uT[2],
        uT[5] + im * uT[6],
        uT[9] + im * uT[10],
        p,
        nt,
    )
    du[2] = QD(
        u[5:8],
        u[1] + im * u[2],
        u[9] + im * u[10],
        uT[1] + im * uT[2],
        uT[5] + im * uT[6],
        uT[9] + im * uT[10],
        p,
        nt,
    )

    du[3] = QD(
        u[9:12],
        u[1] + im * u[2],
        u[5] + im * u[6],
        uT[1] + im * uT[2],
        uT[5] + im * uT[6],
        uT[9] + im * uT[10],
        p,
        nt,
    )
    return collect(Iterators.flatten(du))
end

function QD4(u, E1, E2, ET1, ET2, ET3, ET4, p, nt)
    dF =
        0.5 *
        ((1 - im * p[6]) * (p[2] * (2 * u[3] - 1) - 1)) *
        (u[1] + im * u[2]) +
        im *
        p[7] *
        exp(-im * p[8]) *
        (
            exp(-im * p[9]) * ET1 +
            exp(-im * p[10]) * ET2 +
            exp(-im * p[11]) * ET3 +
            exp(-im * p[12]) * ET4
        ) +
        im * p[12] * (E1 + E2)

    dρ =
        p[4] *
        (FU(u[3], u[4], p[3]) - u[3] - (2 * u[3] - 1) * (u[1]^2 + u[2]^2))

    dn = p[5] * (p[1] - u[4] - 2 * FU(u[3], u[4], p[3]))

    return [real(dF), imag(dF), dρ, dn]
end
function CoupledQD4(u, uT, p, nt, t, t0)

    du = [[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]

    du[1] = QD(
        u[1:4],
        u[5] + im * u[6],
        u[9] + im * u[10],
        uT[1] + im * uT[2],
        uT[5] + im * uT[6],
        uT[9] + im * uT[10],
        uT[12] + im * uT[13],
        p,
        nt,
    )
    du[2] = QD(
        u[5:8],
        u[1] + im * u[2],
        u[9] + im * u[10],
        uT[1] + im * uT[2],
        uT[5] + im * uT[6],
        uT[9] + im * uT[10],
        uT[12] + im * uT[13],
        p,
        nt,
    )

    du[3] = QD(
        u[9:12],
        u[5] + im * u[6],
        u[13] + im * u[14],
        uT[1] + im * uT[2],
        uT[5] + im * uT[6],
        uT[9] + im * uT[10],
        uT[12] + im * uT[13],
        p,
        nt,
    )
    du[4] = QD(
        u[13:16],
        u[5] + im * u[6],
        u[9] + im * u[10],
        uT[1] + im * uT[2],
        uT[5] + im * uT[6],
        uT[9] + im * uT[10],
        uT[12] + im * uT[13],
        p,
        nt,
    )

    return collect(Iterators.flatten(du))
end

## Semi-implicit Euler iteration procedure
# rhs should be the function of the form rhs(u,uT,p,nt,t,t0)
# SIEiters alterates qY, qYhead, qYtail and U
function SIEiters(rhs, num_iter, modstep, T, p, nt, qY1, qY2, qYtail, U, tind)
    # dimensions
    dim = size(qY1, 1)
    # length of the stored history
    hist_len = size(qY1, 2)
    # array with the current intermediate values of the function
    Y1 = zeros(dim)
    Y2 = zeros(dim)
    # them delayed
    YT1 = zeros(dim)
    YT2 = zeros(dim)

    # local variable for the tail
    lqYtail = qYtail[1]

    # initial values of the unknown
    Y1 = qY1[:, lqYtail]

    # local time variables
    t0 = T[1]
    tc = T[2]
    dt = T[3]

    # fixed tau
    ft = floor(Int, p[tind] / dt)

    # RK iterations
    for l = 1:num_iter
        # reading the time-delayed value of variables
        # the queue is stored in an array qY
        # the first index of qY is the number of the integration method subiteration
        # the second index is the point number inside the queue
        # the tail of the queue corresponds to the (t-dt) variables
        di = lqYtail - ft + 1
        if di < 1
            YT1 = qY1[:, hist_len+di]
            YT2 = qY2[:, hist_len+di]
        else
            YT1 = qY1[:, di]
            YT2 = qY2[:, di]
        end

        ntl = nt[l, :]

        # some semi-implicit calculations
        Y2 = Y1 + dt * rhs(Y1, YT1, p, ntl, tc, t0)

        # storing the time-delayed variables
        lqYtail = mod(lqYtail, hist_len) + 1
        qY1[:, lqYtail] = Y1
        qY2[:, lqYtail] = Y2

        # calculate the actual current-time variable
        Y1 = Y1 + dt * rhs(Y2, YT2, p, ntl, tc + dt, t0)

        # integration output
        if mod(l, modstep) == 0
            U[div(l, modstep), :] = Y1
        end

        tc = tc + dt
    end
    T[2] = tc
    qYtail[1] = lqYtail
end

function ArrayGen()
    arr = zeros(dim)
    for i = 1:size(arr,1)
        arr[i] = i
    end
    return arr
end
## our initial history for SIE method subiterations
Random.seed!(1234)
Random.seed!(1234)

qY1 = zeros(dim, hist_len);
qY2 = zeros(dim, hist_len);
for i = 1:hist_len
    qY1[:, i] = randn(dim) .^ 2
    qY1[:, i] = qY1[:, i] ./ maximum(qY1[:, i]) / 1000
    qY2[:, i] = randn(dim) .^ 2
    qY2[:, i] = qY2[:, i] ./ maximum(qY2[:, i]) / 1000
end

## integration
U = zeros(div(num_iter, modstep), dim)

qYtail = [1];

Random.seed!(1234)

time_array = [t0, t0, dt]

# noise terms
nt = zeros(num_iter, 2)

if isQW
    @time SIEiters(
        CoupledQW,
        num_iter,
        modstep,
        time_array,
        p0,
        nt,
        qY1,
        qY2,
        qYtail,
        U,
        tauind,
    )
else
    @time SIEiters(
        CoupledQD,
        num_iter,
        modstep,
        time_array,
        p0,
        nt,
        qY1,
        qY2,
        qYtail,
        U,
        tauind,
    )
end

#
pl_st_t = endtime - 10 * τdelay;
pl_en_t = endtime;
n = 3000; # number of timepoints for plot
pl_st_in = max(1, div(floor(Int, pl_st_t / dt), modstep));
pl_en_in = div(floor(Int, pl_en_t / dt), modstep);
pl_step = max(1, div(floor(Int, (pl_en_in - pl_st_in) / n), modstep));
forplot1 = (
    U[pl_st_in:pl_step:pl_en_in, Reind[1]] .^ 2 +
    U[pl_st_in:pl_step:pl_en_in, Imind[1]] .^ 2
);
ts =
    collect(range(pl_st_t, stop = pl_en_t, length = length(forplot1))) .*
    tphys ./ 10^(-9);
p1 = plot(
    ts,
    forplot1,
    ylims = (0, 1.05 * maximum(forplot1)),
    xlabel = "t",
    ylabel = "Intensity",
    label = "E₁",
    linecolor = :blue,
)

forplot2 = (
    U[pl_st_in:pl_step:pl_en_in, Reind[2]] .^ 2 +
    U[pl_st_in:pl_step:pl_en_in, Imind[2]] .^ 2
);
plot!(
    ts,
    forplot2,
    ylims = (0, 1.05 * maximum(forplot2)),
    xlabel = "t",
    label = "E₂",
    linecolor = :green,
)

forplot3 = (
    U[pl_st_in:pl_step:pl_en_in, Reind[3]] .^ 2 +
    U[pl_st_in:pl_step:pl_en_in, Imind[3]] .^ 2
);
plot!(
    ts,
    forplot3,
    ylims = (0, 1.05 * maximum(forplot3)),
    xlabel = "t",
    label = "E₃",
    linecolor = :red,
)

forplot4 =
    abs2.(
        U[pl_st_in:pl_step:pl_en_in, Reind[1]] .+
        im .* U[pl_st_in:pl_step:pl_en_in, Imind[1]] .+
        U[pl_st_in:pl_step:pl_en_in, Reind[2]] .+
        im .* U[pl_st_in:pl_step:pl_en_in, Imind[2]] .+
        U[pl_st_in:pl_step:pl_en_in, Reind[3]] .+
        im .* U[pl_st_in:pl_step:pl_en_in, Imind[3]],
    )
plot!(
    ts,
    forplot4,
    ylims = (
        0,
        1.05 * maximum(
            collect(
                Iterators.flatten([forplot1, forplot2, forplot3, forplot4]),
            ),
        ),
    ),
    xlabel = "t, ns",
    label = "E₁+E₂+E₃",
    linecolor = :black,
);

pl_st_t = endtime - 800 * τdelay;
pl_en_t = endtime;
pl_st_in = max(1, div(floor(Int, pl_st_t / dt), modstep));
pl_en_in = div(floor(Int, pl_en_t / dt), modstep);

# optical spectrum
optfreq =
    fftshift(
        fftfreq(
            length(U[pl_st_in:pl_en_in, Reind[1]]),
            1 / (dt * modstep * tphys),
        ),
    ) ./ 10^9;

optspec1 =
    10 *
    log10.(
        abs2.(
            fftshift(
                fft(
                    U[pl_st_in:pl_en_in, Reind[1]] +
                    im * U[pl_st_in:pl_en_in, Imind[1]],
                ),
            ),
        ),
    );

optspec2 =
    10 *
    log10.(
        abs2.(
            fftshift(
                fft(
                    U[pl_st_in:pl_en_in, Reind[2]] +
                    im * U[pl_st_in:pl_en_in, Imind[2]],
                ),
            ),
        ),
    );

optspec3 =
    10 *
    log10.(
        abs2.(
            fftshift(
                fft(
                    U[pl_st_in:pl_en_in, Reind[3]] +
                    im * U[pl_st_in:pl_en_in, Imind[3]],
                ),
            ),
        ),
    );

optspec4 =
    10 *
    log10.(
        abs2.(
            fftshift(
                fft(
                    U[pl_st_in:pl_en_in, Reind[1]] .+
                    im .* U[pl_st_in:pl_en_in, Imind[1]] .+
                    U[pl_st_in:pl_en_in, Reind[2]] .+
                    im .* U[pl_st_in:pl_en_in, Imind[2]] .+
                    U[pl_st_in:pl_en_in, Reind[3]] .+
                    im .* U[pl_st_in:pl_en_in, Imind[3]],
                ),
            ),
        ),
    );

optspec0dB = maximum(
    collect(Iterators.flatten([optspec1, optspec2, optspec3, optspec4])),
);
optspec1 = optspec1 .- optspec0dB;
optspec2 = optspec2 .- optspec0dB;
optspec3 = optspec3 .- optspec0dB;
optspec4 = optspec4 .- optspec0dB;

# rf spectrum
rffreq =
    rfftfreq(
        length(U[pl_st_in:pl_en_in, Reind[1]]),
        1 / (dt * modstep * tphys),
    ) ./ 10^9;
rfspec1 =
    10 *
    log10.(
        abs2.(
            rfft(
                U[pl_st_in:pl_en_in, Reind[1]] .^ 2 +
                U[pl_st_in:pl_en_in, Imind[1]] .^ 2,
            ),
        ),
    );
rfspec2 =
    10 *
    log10.(
        abs2.(
            rfft(
                U[pl_st_in:pl_en_in, Reind[2]] .^ 2 +
                U[pl_st_in:pl_en_in, Imind[2]] .^ 2,
            ),
        ),
    );
rfspec3 =
    10 *
    log10.(
        abs2.(
            rfft(
                U[pl_st_in:pl_en_in, Reind[3]] .^ 2 +
                U[pl_st_in:pl_en_in, Imind[3]] .^ 2,
            ),
        ),
    );
rfspec4 =
    10 *
    log10.(
        abs2.(
            rfft(
                abs2.(
                    U[pl_st_in:pl_en_in, Reind[1]] .+
                    im .* U[pl_st_in:pl_en_in, Imind[1]] .+
                    U[pl_st_in:pl_en_in, Reind[2]] .+
                    im .* U[pl_st_in:pl_en_in, Imind[2]] .+
                    U[pl_st_in:pl_en_in, Reind[3]] .+
                    im .* U[pl_st_in:pl_en_in, Imind[3]],
                ),
            ),
        ),
    );

rfspec0dB =
    maximum(collect(Iterators.flatten([rfspec1, rfspec2, rfspec3, rfspec4])));
rfspec1 = rfspec1 .- optspec0dB;
rfspec2 = rfspec2 .- optspec0dB;
rfspec3 = rfspec3 .- optspec0dB;
rfspec4 = rfspec4 .- optspec0dB;

p2 = plot(
    optfreq,
    optspec1,
    xlabel = "Optical Frequency, GHz",
    xlims = (-20, 20),
    ylabel = "Power density, dB",
    legend = false,
    linecolor = :blue,
)

plot!(
    optfreq,
    optspec2,
    xlabel = "Optical Frequency, GHz",
    xlims = (-20, 20),
    ylabel = "Power density, dB",
    legend = false,
    linecolor = :green,
)

plot!(
    optfreq,
    optspec3,
    xlabel = "Optical Frequency, GHz",
    xlims = (-20, 20),
    ylabel = "Power density, dB",
    legend = false,
    linecolor = :red,
)

plot!(
    optfreq,
    optspec4,
    xlabel = "Optical Frequency, GHz",
    xlims = (-10, 20),
    ylabel = "Power density, dB",
    legend = false,
    linecolor = :black,
    ylims = (-100, 5),
)

p3 = plot(
    rffreq,
    rfspec1,
    xlabel = "RF Frequency, GHz",
    xlims = (0, 20),
    ylabel = "Power density, dB",
    legend = false,
    linecolor = :blue,
)

plot!(
    rffreq,
    rfspec2,
    xlabel = "RF Frequency, GHz",
    xlims = (0, 20),
    ylabel = "Power density, dB",
    legend = false,
    linecolor = :green,
)

plot!(
    rffreq,
    rfspec3,
    xlabel = "RF Frequency, GHz",
    xlims = (0, 20),
    ylabel = "Power density, dB",
    legend = false,
    linecolor = :red,
)

plot!(
    rffreq,
    rfspec4,
    xlabel = "RF Frequency, GHz",
    xlims = (0, 20),
    ylims = (-100, 5),
    ylabel = "Power density, dB",
    legend = false,
    linecolor = :black,
)

pall = plot(p1, p2, p3, layout = (3, 1), size = (400, 600), dpi = 300)

display(pall)

# save figure
if isQW
    savefig(pall, string("QW_nJ_", nJthr, "_kap_", κ, "_gam_", γ, ".png"))
else
    savefig(pall, string("QD_nJ_", nJthr, "_kap_", κ, "_gam_", γ, ".png"))
end
