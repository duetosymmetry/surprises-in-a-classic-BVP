from mpmath import *
import numpy as np
from scipy.special import spence

############################################################
## Analytical slopes

def y_prime_B0_asymp(eps):
    return 1 - 3/(2*eps) + log(16)

def one_minus_y_prime_B1_asymp(eps):
    return exp(-3/(2*eps))*(24/eps-16*log(16))

def one_minus_y_prime_M_asymp(eps):
    return exp(-5/(8*eps))*(9/(2*eps)-4*log(4))

############################################################
## Composite results

# atan(1/2) = log(3)/2
atan_half = np.log(3)/2

def asymp_sol_B0_c0(eps, x):
    return x - 2. * np.tanh(x / eps - atan_half)

def asymp_sol_B1_c0(eps, x):
    return -asymp_sol_B0_c0(eps, 1-x)

def asymp_sol_M_c0(eps, x):
    return x - 0.5 - 1.5 * np.tanh(0.75 * (x - 0.5) / eps)

def Li2(z):
    '''SciPy has a nonstandard convention'''
    return spence(1-z)

def asymp_sol_M_c1(eps, x):
    X = (x-0.5)/eps
    sinh = np.sinh(1.5*X)
    cosh = np.cosh(0.75*X)
    log2cosh = np.log(2.*cosh)
    exp = np.exp(-1.5*X)
    log1exp = np.log1p(exp)
    li2 = Li2(-exp)
    
    return -1.5*np.tanh(.75*X) + \
        (eps/(72.*cosh*cosh)) * (4.*np.pi*np.pi + 9.*(8. - 3.*X)*X + 
                                 72.*X*(-log1exp + log2cosh) + 
                                 48.*li2 + 48.*log2cosh*sinh)

def asymp_sol_B0_c1(eps, x):
    c1 = 1. + np.log(12.)
    c2 = 0.125 * (-4.*Li2(-3.) + np.square(np.log(3.)) + 4./3. * np.log(6912.))
    
    X = x/eps
    Xbar = X - atan_half
    
    cosh = np.cosh(Xbar)
    logcosh = np.log(cosh)
    three_exp = 3.*np.exp(-2.*X)
    
    return -2.*np.tanh(Xbar) + \
        0.25*eps/(cosh*cosh) * (4.*c2 + 2*Xbar*(1. + c1 - Xbar - 2.* np.log1p(three_exp) \
                                                + 2. * logcosh) + \
                                2. * Li2(-three_exp) + (-1. + c1 + 2. * logcosh) * np.sinh(2.*Xbar))
def asymp_sol_B1_c1(eps, x):
    return -asymp_sol_B0_c1(eps, 1-x)

############################################################
## Finding eps_c via quadrature

def give_eps_c_integrand(zc):
    def integrand(z):
        arg_of_root = 1-(z+log(1-z))/(zc+log(1-zc))
        return 1/((z-1)*sqrt(arg_of_root))
    return integrand

def eps_c_f_root(zc):
    return zc + log(1-zc) + quad( give_eps_c_integrand(zc), [0, zc] )

def z_c():
    return findroot(eps_c_f_root, -3.9052637703)

def eps_c_of_z_c(zc):
    return -1/(2*(zc+log(1-zc)))

def eps_c():
    return eps_c_of_z_c(z_c())

############################################################
## For computing the period of a solution

def z_min(C, eps):
    return 1 + lambertw(-exp(-1-C*C/(2*eps)), -1)

def z_max(C, eps):
    return 1 + lambertw(-exp(-1-C*C/(2*eps)), 0)

def C_from_y_z(y, z, eps):
    return sqrt(y*y - 2*eps*(z+log(1-z)))

def give_period_integrand(C, eps):
    def integrand(z):
        arg_of_root = C*C + 2*eps*(z+log(1-z))
        return 1/((1-z)*sqrt(arg_of_root))
    return integrand

def period(C, eps):
    return 2*eps*quad( give_period_integrand(C, eps),
                       [z_min(C, eps), z_max(C, eps)])

############################################################
## For some phase space plots

def make_odesys_param_by_arc(eps):
    def odesys(x, state):
        y, z = state

        origdy, origdz = (z, y*(z-1)/eps)
        mag = sqrt(origdy*origdy + origdz*origdz)

        return [origdy/mag, origdz/mag]
    return odesys

def perp_dist_from_point_to_line(p, a, b, c):
    """Compute the perpendicular distance from point
    p=(x0, y0) to the line defined by ax+by+c=0."""
    x0, y0 = p
    return np.abs(a*x0+b*y0+c)/np.sqrt(a*a+b*b)

def add_arrow_on_curve_at_point_closest_to(ys, zs, y0, z0,
                                           i_shift=0, arrowprops=dict(arrowstyle="->")):
    import matplotlib.pyplot as plt
    dists=np.array(list(map(lambda p: sqrt((y0-p[0])*(y0-p[0]) + (z0-p[1])*(z0-p[1])),
                        zip(ys,zs))))
    i_pt = np.argmin(dists) + i_shift
    plt.annotate("", xytext=(ys[i_pt], zs[i_pt]),
                 xy=(ys[i_pt+1], zs[i_pt+1]),
                 arrowprops=arrowprops)

############################################################
## For solving the BVP via shooting

def make_odesys(eps):
    def odesys(x, state):
        y, z = state

        return [z, y*(z-1)/eps]
    return odesys

def y_at_x_eq_1(eps, y_prime_0):
    sol = odefun(make_odesys(eps), 0, [1, y_prime_0])
    y1, _ = sol(1)
    return y1

def one_plus_y_at_x_eq_1(eps, y_prime_0):
    return 1 + y_at_x_eq_1(eps, y_prime_0)

############################################################
## Find initial slopes via shooting

def y_prime_B0_num(eps):
    return findroot(lambda y_prime_0: 1+y_at_x_eq_1(eps, y_prime_0),
                    y_prime_B0_asymp(eps))

def y_prime_B1_num(eps, verbose=False, verify=False):
    delta = one_minus_y_prime_B1_asymp(eps)
    return findroot(lambda y_prime_0: 1+y_at_x_eq_1(eps, y_prime_0), 
                    (1-2*delta, 1), solver='bisect', 
                    verbose=verbose,
                    verify=verify)

def y_prime_M_num(eps, verbose=False, verify=False):
    delta = one_minus_y_prime_M_asymp(eps)
    return findroot(lambda y_prime_0: 1+y_at_x_eq_1(eps, y_prime_0), 
                    (1-2*delta, 1-delta/2), 
                    solver='bisect',
                    verbose=verbose,
                    verify=verify)

