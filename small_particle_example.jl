
file_base = "small_particle_example_"

# load in necessary Julia packages
using Statistics
using Profile
using PyCall
using HDF5
using TickTock
import Base.Threads.@spawn

# load in necessary Python packages (used for python fast marching implementations)
skfmm = pyimport("skfmm")
np = pyimport("numpy")

# set contants
const BANDTHICKNESS = 5
const EPSILON = 1e-4
const sub_offset = 10

# load in anisotropic surface energy
xisData=h5open("xis_DFT_Ni_interp_100_out_45_deg_rot_x_axis_0995_005_rounded_edges_r_-0_1_even_denser.HDF5","r") do file
  read(file)
end

xis = xisData["Dataset1"]
xis_x_lookup = copy(xis[1,:])
xis_y_lookup = copy(xis[2,:])
xis_z_lookup = copy(xis[3,:])

# load in anisotropic diffusivities 
diffsData=h5open("diffs_DFT_Ni_interp_100_out_45deg_rot_x_axis_T_01_even_denser.HDF5","r") do file
  read(file)
end

diffs = diffsData["Dataset1"]
global diffs_lookup = copy(diffs)

# code for parsing spherical lookup table, as dissussed in Smith et al. 2012 
# J. Smith, G. Petrova, and S. Schaefer, Encoding Normal Vectors Using Optimized Spherical Coordinates, 
# in Computers and Graphics (Pergamon), Vol. 36 (Elsevier Ltd, 2012), pp. 360–365.

const eps = acos(1-0.000001)

function N_theta(j,Nphi,eps)
  if j == 0 || j == Nphi-1
    return 1.
  end
  phi = j*pi/(Nphi-1.)
  denom_num = cos(eps) - cos(phi)*cos(phi+pi/(2*(Nphi-1)))
  denom_denom = sin(phi)*sin(phi+pi/(2*(Nphi-1)))
  return ceil(pi/acos(denom_num/denom_denom))
end

function j_thetas(N_theta)
  N_theta = convert(Int,N_theta)
  thetas = zeros(Float64,N_theta)
  for k in 0:N_theta -1
    thetas[k+1] = k*2*pi/N_theta
  end
  thetas
end

global all_thetas = Array{Array{T,1} where T,1}[]
Nphi = 1600
Nths = zeros(convert(Int,Nphi))
for j in 0:Nphi-1
  Nthj = N_theta(j,Nphi,eps)
  Nths[j+1] = Nthj
  global all_thetas = vcat(all_thetas,[j_thetas(Nthj)])
end

global Nths_cum = cumsum(Nths)
Nths_cum = vcat(0.,Nths_cum)
Nths_cum = convert(Array{Int},Nths_cum)

# code for taking narrow band derivatives
function band_dx(arr,band)
  new_arr = zeros(size(arr))
  a,b,c = size(arr)
  temp_arr = zeros(a+1,b,c)
  temp_arr[1:end-1,:,:] .= arr
  temp_arr[end,:,:] .= arr[end,:,:]
  shift = CartesianIndex(1,0,0)
  for point in band
    new_arr[point] = (temp_arr[point+shift]-temp_arr[point-shift])/2
  end
  new_arr[end,:,:] .= 2 .* new_arr[end,:,:]
  new_arr
end


function band_d2x(arr,band)
  new_arr = zeros(size(arr))
  a,b,c = size(arr)
  temp_arr = zeros(a+1,b,c)
  temp_arr[1:end-1,:,:] .= arr
  temp_arr[end,:,:] .= 2 .* arr[end,:,:] .- arr[end-1,:,:]
  shift = CartesianIndex(1,0,0)
  for point in band
    new_arr[point] = temp_arr[point+shift]+temp_arr[point-shift]- 2 * temp_arr[point]
  end
  dx = band_dx(arr,band)
  dx_dx = band_dx(dx,band)
  new_arr[end-2:end,:,:] .= dx_dx[end-2:end,:,:]
  new_arr
end

function band_dy(arr,band)
  new_arr = zeros(size(arr))
  shift = CartesianIndex(0,1,0)
  for point in band
    new_arr[point] = (arr[point+shift]-arr[point-shift])/2
  end
  new_arr
end

function band_d2y(arr,band)
  new_arr = zeros(size(arr))
  shift = CartesianIndex(0,1,0)
  for point in band
    new_arr[point] = arr[point+shift]+arr[point-shift]- 2 * arr[point]
  end
  new_arr
end

function band_dz(arr,band)
  new_arr = zeros(size(arr))
  shift = CartesianIndex(0,0,1)
  for point in band
    new_arr[point] = (arr[point+shift]-arr[point-shift])/2
  end
  new_arr
end


function band_d2z(arr,band)
  new_arr = zeros(size(arr))
  shift = CartesianIndex(0,0,1)
  for point in band
    new_arr[point] = arr[point+shift]+arr[point-shift]- 2 * arr[point]
  end
  new_arr
end


function band_grad(arr,band)
  dx = band_dx(arr,band)
  dy = band_dy(arr,band)
  dz = band_dz(arr,band)
  [dx,dy,dz]
end


function s_derivative(grad_field,nx,ny,nz,band)
  nx = nx[band]
  ny = ny[band]
  nz = nz[band]
  dx = grad_field[1][band]
  dy = grad_field[2][band]
  dz = grad_field[3][band]
  ndnk_x = nx.*dx
  ndnk_y = ny.*dy
  ndnk_z = nz.*dz
  ndnk_scale = ndnk_x.+ndnk_y.+ndnk_z
  new_arr = [zeros(size(grad_field[1])) for i in 1:3]
  new_arr[1][band] = dx.-ndnk_scale.*nx
  new_arr[2][band] = dy.-ndnk_scale.*ny
  new_arr[3][band] = dz.-ndnk_scale.*nz
  new_arr
end

function ss_derivative(s_field,nx,ny,nz,band)
  A_x,A_y,A_z = band_grad(s_field[1],band)
  Bs= @spawn band_grad(s_field[2],band)
  C_x,C_y,C_z = band_grad(s_field[3],band)
  get_arr = @spawn zeros(size(nx))
  B_x,B_y,B_z  = fetch(Bs)
  new_arr = fetch(get_arr)
  Threads.@threads for loc in band
    new_arr[loc] = A_x[loc]+B_y[loc]+C_z[loc]-nx[loc]*(nx[loc]*A_x[loc]+ny[loc]*A_y[loc]+nz[loc]*A_z[loc])-
          ny[loc]*(nx[loc]*B_x[loc]+ny[loc]*B_y[loc]+nz[loc]*B_z[loc])-
          nz[loc]*(nx[loc]*C_x[loc]+ny[loc]*C_y[loc]+nz[loc]*C_z[loc])
  end
  new_arr
end

function ss_derivative_flux(s_field,nx,ny,nz,band)
  A_x,A_y,A_z = band_grad(s_field[1],band)
  Bs= @spawn band_grad(s_field[2],band)
  C_x,C_y,C_z = band_grad(s_field[3],band)
  get_arr = @spawn zeros(size(nx))
  B_x,B_y,B_z  = fetch(Bs)
  new_arr = fetch(get_arr)
  sub_arr = zeros(size(nx))
  Threads.@threads for loc in band
    new_arr[loc] = A_x[loc]+B_y[loc]+C_z[loc]-nx[loc]*(nx[loc]*A_x[loc]+ny[loc]*A_y[loc]+nz[loc]*A_z[loc])-
          ny[loc]*(nx[loc]*B_x[loc]+ny[loc]*B_y[loc]+nz[loc]*B_z[loc])-
          nz[loc]*(nx[loc]*C_x[loc]+ny[loc]*C_y[loc]+nz[loc]*C_z[loc])

    sub_arr[loc] = C_z[loc]+B_y[loc]-nz[loc]*(nz[loc]*C_z[loc]+ny[loc]*C_y[loc])-
          ny[loc]*(nz[loc]*B_z[loc]+ny[loc]*B_y[loc])
  end
  new_arr[end,:,:] .= sub_arr[end,:,:] .+ 0.5 .* (new_arr[end,:,:] .- sub_arr[end,:,:])
  new_arr
end

function ss_second_order(phi,nx,ny,nz,band)
  d2x = band_d2x(phi,band)
  d2y = band_d2y(phi,band)
  d2z = band_d2z(phi,band)
  return(d2x .+ d2y .+ d2z)
end 


function H_sussman(phi)
  output = zeros(size(phi))
  for (i,p) in enumerate(phi)
    if p > 1
      output[i] = 1
    elseif p < -1
      output[i] = 0
    else
      output[i] = 0.5*(1+p+sin(pi*p)/pi)
    end
  end
  output
end

function H_prime_sussman(phi)
  output = zeros(size(phi))
  for (i,p) in enumerate(phi)
    if -1<p<1
      output[i] = (1+cos(pi*p))/2
    end
  end
  output
end

function band_dx_plus!(new_arr,arr,band)
  shift = CartesianIndex(1,0,0)
  a,b,c = size(arr)
  temp_arr = zeros(a+1,b,c)
  temp_arr[1:end-1,:,:] .= arr
  temp_arr[end,:,:] .= arr[end,:,:]
  Threads.@threads for point in band
    new_arr[point] = (temp_arr[point+shift]-temp_arr[point])
  end
  new_arr[end,:,:] .= new_arr[end-1,:,:]
  new_arr
end

function band_dx_minus!(new_arr,arr,band)
  shift = CartesianIndex(1,0,0)
  Threads.@threads for point in band
    new_arr[point] = (arr[point]-arr[point-shift])
  end
  new_arr
end

function band_dy_plus!(new_arr,arr,band)
  shift = CartesianIndex(0,1,0)
  Threads.@threads for point in band
    new_arr[point] = (arr[point+shift]-arr[point])
  end
  new_arr
end

function band_dy_minus!(new_arr,arr,band)
  shift = CartesianIndex(0,1,0)
  Threads.@threads for point in band
    new_arr[point] = (arr[point]-arr[point-shift])
  end
  new_arr
end

function band_dz_plus!(new_arr,arr,band)
  shift = CartesianIndex(0,0,1)
  Threads.@threads for point in band
    new_arr[point] = (arr[point+shift]-arr[point])
  end
  new_arr
end

function band_dz_minus!(new_arr,arr,band)
  shift = CartesianIndex(0,0,1)
  Threads.@threads for point in band
    new_arr[point] = (arr[point]-arr[point-shift])
  end
  new_arr
end

function sign_sussman(phi)
  2 .*(H_sussman(phi).-0.5)
end


function Dx_sussman!(new_arr,d_plus,d_minus,phi,sign_phi,band)
  band_dx_plus!(d_plus,phi,band)
  band_dx_minus!(d_minus,phi,band)
  Threads.@threads for point in band
    if (d_plus[point]*sign_phi[point]) <0 && (d_minus[point]+d_plus[point]) * sign_phi[point] < 0
      new_arr[point] = d_plus[point]
    elseif (d_minus[point]*sign_phi[point])>0 && (d_plus[point]+d_minus[point])*sign_phi[point] > 0
      new_arr[point] = d_minus[point]
    end
  end
  new_arr
end

function Dy_sussman!(new_arr,d_plus,d_minus,phi,sign_phi,band)
  band_dy_plus!(d_plus,phi,band)
  band_dy_minus!(d_minus,phi,band)
  Threads.@threads for point in band
    if (d_plus[point]*sign_phi[point]) <0 && (d_minus[point]+d_plus[point]) * sign_phi[point] < 0
      new_arr[point] = d_plus[point]
    elseif (d_minus[point]*sign_phi[point])>0 && (d_plus[point]+d_minus[point])*sign_phi[point] > 0
      new_arr[point] = d_minus[point]
    end
  end
  new_arr
end

function Dz_sussman!(new_arr,d_plus,d_minus,phi,sign_phi,band)
  band_dz_plus!(d_plus,phi,band)
  band_dz_minus!(d_minus,phi,band)
  Threads.@threads for point in band
    if (d_plus[point]*sign_phi[point]) <0 && (d_minus[point]+d_plus[point]) * sign_phi[point] < 0
      new_arr[point] = d_plus[point]
    elseif (d_minus[point]*sign_phi[point])>0 && (d_plus[point]+d_minus[point])*sign_phi[point] > 0
      new_arr[point] = d_minus[point]
    end
  end
  new_arr
end


function grad_norm_sussman!(new_arr,dx,dy,dz,scratch1,scratch2,phi,sign_phi,band)
  Dx_sussman!(dx,scratch1,scratch2,phi,sign_phi,band)
  Dy_sussman!(dy,scratch1,scratch2,phi,sign_phi,band)
  Dz_sussman!(dz,scratch1,scratch2,phi,sign_phi,band)
  Threads.@threads for ind in band #eachindex(new_arr)
    new_arr[ind] = (dx[ind]^2 + dy[ind]^2 + dz[ind]^2)^0.5
  end
end

function L_sussman!(L,phi,sign_phi,dx,dy,dz,scratch1,scratch2,
    grad_norm_temp,band)
  grad_norm_sussman!(grad_norm_temp,dx,dy,dz,scratch1,scratch2,phi,sign_phi,band)
  Threads.@threads for ind in band #eachindex(phi)
    L[ind] = sign_phi[ind] * (1 - grad_norm_temp[ind])
  end
end

function phi_temp_sussman!(phi_temp,phi,sign_phi,band,dx,dy,dz,scratch1,scratch2,scratch3,
    grad_norm_temp,dt=0.5)
  L_sussman!(scratch3,phi,sign_phi,dx,dy,dz,scratch1,scratch2,
    grad_norm_temp,band)
  Threads.@threads for ind in band #eachindex(phi)
    phi_temp[ind] = phi[ind] + dt * scratch3[ind]
  end
end

function integrate_domain(phi,i,j,k,xmax,ymax,zmax)
  denom = 51
  integral = 51*phi[i,j,k]
  for ii in max(1,i-1):min(xmax,i+1), jj in max(1,j-1):min(ymax,j+1), kk in
    max(1,k-1):min(zmax,k+1)
    denom += 1
    integral += phi[ii,jj,kk]
  end
  integral = integral/denom
end
 
function integral_sussman!(new_arr,phi,band)
  a,b,c = size(phi)
  Threads.@threads for point in band
    i,j,k = Tuple(point)
    new_arr[point] = integrate_domain(phi,i,j,k,a,b,c)
  end
  new_arr
end

function lambda_sussman!(lambda,phi_0,phi_temp,grad_norm,hps,band,num_input,denom_input,numerator,denominator,dt = 0.5)
  Threads.@threads for ind in band #eachindex(phi_temp)
    num_input[ind] = hps[ind]*(phi_temp[ind]-phi_0[ind])/dt
    denom_input[ind] = hps[ind]^2 * grad_norm[ind]
  end
  integral_sussman!(numerator,num_input,band)
  integral_sussman!(denominator,denom_input,band) 
  Threads.@threads for ind in band #eachindex(phi_temp)
    lambda[ind] = -numerator[ind]/(denominator[ind]+ 1e-5)
  end
  lambda
end

function iterate_sussman!(phi,phi_0,phi_temp,sign_phi,hps,band,dx,dy,dz,scratch1,scratch2,
    scratch3,scratch4,scratch5,grad_norm,grad_norm_temp,dt = 0.5)
  phi_temp_sussman!(phi_temp,phi,sign_phi,band,dx,dy,dz,scratch1,scratch2,scratch3,
    grad_norm_temp,dt)
  lambda_sussman!(scratch5,phi_0,phi_temp,grad_norm,hps,band,scratch1,scratch2,scratch3,scratch4,dt)
  Threads.@threads for ind in band #eachindex(phi)
    phi[ind] = phi_temp[ind] + dt*scratch5[ind]*hps[ind]*grad_norm[ind]
  end
end


function fast_volume(phi)
  sum(H_sussman(-phi))
end


const pad_length = 5
function repad_phi!(phi)
# this is where any boundary conditions are implemented
  phi
end


function redistance_periodic(phi,num_its,band,dt=0.01)
  a,b,c = size(phi)
  extension = reshape(phi[end,:,:] .+ phi[end,:,:] .- phi[end-1,:,:],1,b,c)
  phi = cat(phi,extension,dims=1)
  phi_0 = copy(phi)
  mask = BitArray(undef,size(phi))
  mask .= false
  mask[2:end,2:end-1,2:end-1] .= true
  bottom = BitArray(undef,size(phi))
  bottom .= false
  bottom[end-2:end,2:end-1,2:end-1] .= true
  in_band = BitArray(undef,size(phi))
  in_band .= false
  for loc in band
    in_band[loc] = true
  end
  in_band .= in_band .| ((phi .< BANDTHICKNESS+1) .& mask) .| bottom
  band = convert(Array{CartesianIndex{3},1},findall(in_band))
  
  dx = zeros(size(phi))
  dy = zeros(size(phi))
  dz = zeros(size(phi))
  phi_temp = zeros(size(phi))
  scratch1 = zeros(size(phi))
  scratch2 = zeros(size(phi))
  scratch3 = zeros(size(phi))
  scratch4 = zeros(size(phi))
  scratch5 = zeros(size(phi))
  grad_norm = zeros(size(phi))
  grad_norm_temp = zeros(size(phi))
  phi_0 = copy(phi)
  sign_phi = sign_sussman(phi_0)
  grad_norm_sussman!(grad_norm,dx,dy,dz,scratch1,scratch2,phi_0,sign_phi,band)
  hps = H_prime_sussman(phi_0)
  for i in 1:num_its
    dt = 0.8*dt
    iterate_sussman!(phi,phi_0,phi_temp,sign_phi,hps,band,dx,dy,dz,scratch1,scratch2,
    scratch3,scratch4,scratch5,grad_norm,grad_norm_temp,dt)
    repad_phi!(phi)
  end
  phi[1:end-1,:,:]
end

# Make initial phi
wire = 0.5 .*ones(20,40,40)
a,b,c = size(wire)
wire[16:end,10:30,10:30] .= -0.5
hole = ones(size(wire))
repad_phi!(wire)
phi = skfmm.distance(wire)




mask = BitArray(undef,size(phi))
mask .= false
mask[:,2:end-1,2:end-1] .= true
repad_phi!(phi)
phi_dist = skfmm.distance(phi,narrow=BANDTHICKNESS+1)
zero_band = convert(Array{CartesianIndex{3},1},findall((phi == 0) .& mask))
two_band = cat(findall((0 .< abs.(phi_dist) .< 3).& mask),zero_band,dims=1)
band = cat(findall((0 .< abs.(phi_dist) .<= BANDTHICKNESS) .& mask),zero_band,dims=1)

phi = redistance_periodic(phi,200,band)
repad_phi!(phi)

np.save(string(file_base,"0.npy"),phi[1:end-1,:,:])






function buffer_phi(phi::Array{Float64,2})::Array{Float64,2}
  return(cat(phi,phi[end,:]',dims=1))
end

function get_one_band(phi::Array{Float64,3},
band::Array{CartesianIndex{3},1})::Array{CartesianIndex{3},1}
  one_band = []
  shiftx = CartesianIndex(1,0,0)
  shifty = CartesianIndex(0,1,0)
  shiftz = CartesianIndex(0,0,1)
  a,b,c = size(phi)
  for loc in band
    prods = [phi[loc]*phi[min(loc[1]+1,a),loc[2],loc[3]],phi[loc]*phi[loc-shiftx],
    phi[loc]*phi[loc+shifty],phi[loc]*phi[loc-shifty],
    phi[loc]*phi[loc+shiftz],phi[loc]*phi[loc-shiftz]]
    if minimum(prods) <= 0
      append!(one_band,tuple(loc))
    end
  end
  one_band = convert(Array{CartesianIndex{3},1},one_band)
  return(one_band)
end

function get_wmc(phi::Array{Float64,3},nx::Array{Float64,3},ny::Array{Float64,3},nz,grad_norm,
band::Array{CartesianIndex{3},1},one_band::Array{CartesianIndex{3},1},sub_two_band,xis_x_lookup,xis_y_lookup,xis_z_lookup)::Array{Float64,3}
  wmc = zeros(size(phi))
  phis = acos.(nz[band])
  thetas = (atan.(ny[band],nx[band]).+2*pi).%(2*pi)
  js = convert(Array{Int,1},round.(phis.*(Nphi-1)./pi))
  Nthjs = Nths[js.+1]
  ks = convert(Array{Int,1},round.(thetas.*Nthjs./(2*pi)).%Nthjs)
  locs = Nths_cum[js.+1].+ks.+1
  xi_x = zeros(size(nx))
  xi_y = zeros(size(ny))
  xi_z = zeros(size(nz))  
  xi_x[band] = xis_x_lookup[locs]
  xi_y[band] = xis_y_lookup[locs]
  xi_z[band] = xis_z_lookup[locs]
  plane_norm = sqrt.(ny.^2 .+ nz.^2 .+ EPSILON)
  c_Gamma_x = sqrt.(plane_norm.^2 ./ grad_norm.^2)
  c_Gamma_y = - nx .* ny .* c_Gamma_x ./ plane_norm.^2
  c_Gamma_z = -nx .* nz .* c_Gamma_x ./ plane_norm.^2
  xi_dot_n = xi_x.*nx .+ xi_y .*ny .+ xi_z .* nz
  xi_dot_c_Gamma = xi_x .* c_Gamma_x .+ xi_y .* c_Gamma_y .+ xi_z .* c_Gamma_z
  c_gg_x = xi_dot_n .* c_Gamma_x .- xi_dot_c_Gamma .* nx
  c_gg_y = xi_dot_n .* c_Gamma_y .- xi_dot_c_Gamma .* ny
  c_gg_z = xi_dot_n .* c_Gamma_z .- xi_dot_c_Gamma .* nz
  wmc[one_band] .= ss_derivative([xi_x,xi_y,xi_z],nx,ny,nz,band)[one_band]
  trash, wmc_temp = skfmm.extension_velocities(phi,wmc,narrow = BANDTHICKNESS)
  phi_div = ss_second_order(phi,nx,ny,nz,band)
  k = ss_derivative([nx,ny,nz],nx,ny,nz,band)
  grad_k = band_grad(k,band)
  k_s = s_derivative(grad_k,nx,ny,nz,band)
  k_ss = ss_derivative(k_s,nx,ny,nz,band)
  willmore = k_ss .+ (k.^3)./2
  n_Gamma_y = ny./plane_norm
  n_Gamma_z = nz./plane_norm
  wmc_temp[sub_two_band] .= wmc_temp[sub_two_band] .+ (c_gg_y[sub_two_band] .* n_Gamma_y[sub_two_band] .+ c_gg_z[sub_two_band] .* n_Gamma_z[sub_two_band] .+ 0.6) .* plane_norm[sub_two_band] ./ sqrt.(1 .+ nx[sub_two_band].^2)
  wmc_temp .= wmc_temp .+ 0.1 .* phi_div .- 1e0 .* willmore
  trash,wmc = skfmm.extension_velocities(phi,wmc_temp,narrow = BANDTHICKNESS)
  return(wmc)
end

function S_extender(phi,k_ss,grad_norm,one_band)
  S = zeros(size(phi))
  S[one_band] = -grad_norm[one_band].*k_ss[one_band]
  trash,S = skfmm.extension_velocities(phi,S,narrow=BANDTHICKNESS)
  return(S)
end

function get_S(phi::Array{Float64,3},nx::Array{Float64,3},ny::Array{Float64,3},
nz,wmc::Array{Float64,3},grad_norm::Array{Float64,3},band::Array{CartesianIndex{3},1},
one_band::Array{CartesianIndex{3},1},sub_two_band)::Array{Float64,3}
  diffs = ones(size(phi))
  phis = acos.(nz[band])
  thetas = (atan.(ny[band],nx[band]).+2*pi).%(2*pi)
  js = convert(Array{Int,1},round.(phis.*(Nphi-1)./pi))
  Nthjs = Nths[js.+1]
  ks = convert(Array{Int,1},round.(thetas.*Nthjs./(2*pi)).%Nthjs)
  locs = Nths_cum[js.+1].+ks.+1
  diffs[band] .= diffs_lookup[locs]
  grad_k = band_grad(wmc,band)
  k_s = s_derivative(grad_k,nx,ny,nz,band)
  k_s[1][sub_two_band] .= 0
  k_s[2][sub_two_band] .= 0
  k_s[3][sub_two_band] .= 0
  k_ss = ss_derivative([k_s[1].*diffs,k_s[2].*diffs,k_s[3].*diffs],
    nx,ny,nz,band)
  S = S_extender(phi,k_ss,grad_norm,one_band)
  return(S)
end

function step_sim(phi::Array{Float64,3},xis_x_lookup,xis_y_lookup,alpha)
  mask = BitArray(undef,size(phi))
  mask .= false
  mask[:,2:end-1,2:end-1] .= true
  repad_phi!(phi)
  phi_dist = skfmm.distance(phi,narrow=BANDTHICKNESS+1)
  zero_band = convert(Array{CartesianIndex{3},1},findall((phi == 0) .& mask))
  two_band = cat(findall((0 .< abs.(phi_dist) .< 3).& mask),zero_band,dims=1)
  band = cat(findall((0 .< abs.(phi_dist) .<= BANDTHICKNESS) .& mask),zero_band,dims=1)
  one_band = get_one_band(phi,band)
  temp = 100*ones(size(phi))
  temp[end-1,:,:] = skfmm.distance(phi[end-1,:,:])
  temp[end,:,:] = temp[end-1,:,:]
  template = abs.(temp[end-1,:,:]) .< 3
  sub_two_band = convert(Array{CartesianIndex{3},1},findall(abs.(temp) .< 3))
  bottom = phi[end-1,:,:] .+ (phi[end-1,:,:] .- phi[end-2,:,:])
  phi[end,:,:] .= template .* bottom .+ .!template .* phi[end,:,:]
  nx = zeros(Float64,size(phi))
  ny = zeros(Float64,size(phi))
  nz = zeros(Float64,size(phi))
  phi_x,phi_y,phi_z = band_grad(phi,band)
  grad_norm = (EPSILON.+phi_x.^2+phi_y.^2+phi_z.^2).^0.5
  nx[band] = phi_x[band]./grad_norm[band]
  ny[band] = phi_y[band]./grad_norm[band]
  nz[band] = phi_z[band]./grad_norm[band]
  wmc = get_wmc(phi,nx,ny,nz,grad_norm,two_band,one_band,sub_two_band,xis_x_lookup,xis_y_lookup,xis_z_lookup)
  S = get_S(phi,nx,ny,nz,wmc,grad_norm,two_band,one_band,sub_two_band)
  dt_out =  alpha/ maximum(abs.(S[:,pad_length+1:end-pad_length,pad_length+1:end-pad_length]))
  phi .= (phi .+ dt_out .* S)
  return(phi,dt_out,band)
end

function iterate_sim(phi_init::Array{Float64,3},file_base::String,
num_its::Int,xis_x_lookup,xis_y_lookup)
  phi = copy(phi_init)
  time_file = string(file_base,"times.txt")
  dist_file = string(file_base,"distance.txt")
  tot_time = 0.
  volume_init = fast_volume(phi[1:end-1,:,:])
  alpha_base = 0.01
  kp = 1e0
  ki = 0.001
  kd = 5e1
  dV = 0
  d_dV = 0
  for i in 1:num_its
    
    if i%10 == 0
      println(dV)
      needs_redist = false
      phi_dist = skfmm.distance(phi)
      println(minimum(phi_dist[1,:,:]))
      if minimum(phi_dist[1,:,:]) < (BANDTHICKNESS + 2)
        a,b,c = size(phi)
        phi = cat(4*BANDTHICKNESS*ones(BANDTHICKNESS,b,c),phi,dims = 1)
        needs_redist = true
        println("expand")
        println(size(phi))
      end
      if minimum(phi_dist[:,1,:]) < (BANDTHICKNESS + 2)
        a,b,c = size(phi)
        phi = cat(4*BANDTHICKNESS*ones(a,BANDTHICKNESS,c),phi,dims = 2)
        needs_redist = true
        println("expand")
        println(size(phi))
      end
      if minimum(phi_dist[:,end,:]) < (BANDTHICKNESS + 2)
        a,b,c = size(phi)
        phi = cat(phi,4*BANDTHICKNESS*ones(a,BANDTHICKNESS,c),dims = 2)
        needs_redist = true
        println("expand")
        println(size(phi))
      end
      if minimum(phi_dist[:,:,1]) < (BANDTHICKNESS + 2)
        a,b,c = size(phi)
        phi = cat(4*BANDTHICKNESS*ones(a,b,BANDTHICKNESS),phi,dims = 3)
        needs_redist = true
        println("expand")
        println(size(phi))
      end
      if minimum(phi_dist[:,:,end]) < (BANDTHICKNESS + 2)
        a,b,c = size(phi)
        phi = cat(phi,4*BANDTHICKNESS*ones(a,b,BANDTHICKNESS),dims = 3)
        needs_redist = true
        println("expand")
        println(size(phi))
      end
      if needs_redist
        mask = BitArray(undef,size(phi))
        mask .= false
        mask[:,2:end-1,2:end-1] .= true
        repad_phi!(phi)
        phi_dist = skfmm.distance(phi)
        zero_band = convert(Array{CartesianIndex{3},1},findall((phi == 0) .& mask))
        two_band = cat(findall((0 .< abs.(phi_dist) .< 3).& mask),zero_band,dims=1)
        band = cat(findall((0 .< abs.(phi_dist) .<= BANDTHICKNESS) .& mask),zero_band,dims=1)
        Threads.@threads for loc in band
          phi_dist[loc] = phi[loc]
        end
        phi = copy(phi_dist)
        phi = redistance_periodic(phi,20,band)
        np.save(string(file_base,"repad_$(i).npy"),phi)
        println("repad_$(i)")
      end
    end
    if i > 400
      alpha_base = max(min(alpha_base - kd * d_dV,0.1),0.001)
    end
    phi,dt, band = step_sim(phi,xis_x_lookup,xis_y_lookup,alpha_base)
    phi = redistance_periodic(phi,20,band)
    Threads.@threads for loc in eachindex(phi)
      if abs(phi[loc]) > 4*BANDTHICKNESS
        phi[loc] = 4*BANDTHICKNESS * sign(phi[loc])
      end
    end
    volume = fast_volume(phi[1:end-1,:,:])
    dV_last = copy(dV)
    dV = volume/volume_init - 1
    d_dV = dV - dV_last
    tot_time = tot_time + dt


    if i%100 == 0
      println(fast_volume(phi[1:end-1,:,:]))
      println(alpha_base)
      println(dV)
      println(size(phi))
      println(size(phi_init))
      np.save(string(file_base,"$(i).npy"),phi[1:end-1,:,:])
      open(time_file,"a") do f
        write(f,"$(tot_time)\n")
      end
      println(i)
    end
  end
  println("done")
end
tick()
iterate_sim(phi,file_base,20,xis_x_lookup,xis_y_lookup)
tock()
tick()
iterate_sim(phi,file_base,20,xis_x_lookup,xis_y_lookup)
tock()
tick()
iterate_sim(phi,file_base,50000,xis_x_lookup,xis_y_lookup)
tock()

