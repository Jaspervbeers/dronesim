'''
Utility functions for control systems analysis in Python
'''
import numpy as np
import control as c
import matplotlib.pyplot as plt
from funcs import plotting

def _svd(sys, omega = np.logspace(-4, 4, 1000)):
    G = sys(omega * 1j)
    singular_values = [
        np.linalg.svd(G[..., k], compute_uv=False) for k in range(len(omega))
    ]
    return np.vstack(singular_values), omega


def svd_full(sys, omega = np.logspace(-4, 4, 1000)):
    G = sys(omega * 1j)
    singular_values = []
    Us = []
    VTs = []
    for k in range(len(omega)):
        u, s, v = np.linalg.svd(G[..., k])
        singular_values.append(s)
        Us.append(u)
        VTs.append(v)
    return np.vstack(singular_values), omega, np.array(Us), np.array(VTs)


def svd(sys, omega = np.logspace(-4, 4, 1000)):
    sigma = _svd(sys, omega=omega)
    sigma_max = sigma.max()
    sigma_min = sigma.min()
    condition_number = sigma_max/sigma_min
    return omega, sigma, [sigma_min, sigma_max], condition_number


def bode(G, omega = np.logspace(-2, 4, 2000), parentFig = None, returnFig = False, color = 'tab:blue'):
    if parentFig is None:
        fig = plt.figure()
        gs = fig.add_gridspec(2*G.noutputs, G.ninputs)
    else:
        fig = parentFig
        if len(fig.axes) != 2*G.noutputs*G.ninputs:
            raise ValueError(f'Not enough axes in parent figure, expected {2*G.noutputs*G.ninputs} but got {len(fig.axes)}')
    
    mags = []
    phases = []
    axesG = []
    axesP = []
    counter = 0
    for o in range(G.noutputs):
        mags_o = []
        phases_o = []
        axesG_o = []
        axesP_o = []
        for i in range(G.ninputs):
            if parentFig is None:
                if i > 0:
                    if o > 0:
                        axG = fig.add_subplot(gs[2*o, i], sharey = axesG_o[0], sharex = axesG[0][i])
                    else:
                        axG = fig.add_subplot(gs[2*o, i], sharey = axesG_o[0])
                    axP = fig.add_subplot(gs[2*o + 1, i], sharey = axesP_o[0], sharex = axG)
                else:
                    if o > 0:
                        axG = fig.add_subplot(gs[2*o, i], sharex = axesG[0][i])
                    else:
                        axG = fig.add_subplot(gs[2*o, i])
                    axP = fig.add_subplot(gs[2*o + 1, i], sharex = axG)
            else:
                axG = fig.axes[counter]
                axP = fig.axes[counter + 1]
            mag, phase, _ = c.bode(G[o, i], omega=omega, plot=False)
            axG.semilogx(omega, c.mag2db(mag), linewidth = 2, color = color)
            axP.semilogx(omega, phase*180/np.pi, linewidth = 2, color = color)
            axG.grid('on')
            axP.grid('on')
            if parentFig is None:
                if i == 0:
                    axG.set_ylabel(r'$|\cdot|$, dB')
                    axP.set_ylabel(r'$\angle \cdot$, deg')
                else:
                    plt.setp(axG.get_yticklabels(), visible = False)
                    plt.setp(axP.get_yticklabels(), visible = False)
                if o == G.noutputs - 1:
                    axP.set_xlabel('Frequency, rad/s')
                else:
                    plt.setp(axP.get_xticklabels(), visible = False)
                plt.setp(axG.get_xticklabels(), visible = False)
            mags_o.append(mag)
            axesG_o.append(axG)
            phases_o.append(phase)
            axesP_o.append(axP)
            counter += 2
        mags.append(mags_o)
        phases.append(phases_o)
        axesG.append(axesG_o)
        axesP.append(axesP_o)
    # plt.tight_layout()
    if returnFig:
        return mags, phases, omega, fig
    else:
        return mags, phases, omega


def removeNearZero(tf, rtol=1e-05, atol=1e-08):
    nums = []
    dens = []
    for o in range(tf.noutputs):
        nums_o = []
        dens_o = []
        for j in range(tf.ninputs):
            nums_o.append([i if not np.isclose(i, 0, rtol = rtol, atol=atol) else 0 for i in tf.num[o][j]])
            dens_o.append([i if not np.isclose(i, 0, rtol = rtol, atol=atol) else 0 for i in tf.den[o][j]])
        nums.append(nums_o)
        dens.append(dens_o)    
    return c.TransferFunction(nums, dens, tf.dt)


def minreal(TF, tol=None, debug = False):
        """
        Remove cancelling pole/zero pairs from a transfer function
        
        **ADAPTED FROM THE PYTHON CONTROL TOOLBOX**
            -> Instead of cancelling first close pole-zero pair, we now 
            cancel the closest pair
        """
        # based on octave minreal

        # default accuracy
        from sys import float_info
        sqrt_eps = np.sqrt(float_info.epsilon)

        # pre-allocate arrays
        num = [[[] for j in range(TF.ninputs)] for i in range(TF.noutputs)]
        den = [[[] for j in range(TF.ninputs)] for i in range(TF.noutputs)]

        for i in range(TF.noutputs):
            for j in range(TF.ninputs):

                # split up in zeros, poles and gain
                newzeros = []
                zeros = np.roots(TF.num[i][j])
                poles = np.roots(TF.den[i][j])
                gain = TF.num[i][j][0] / TF.den[i][j][0]

                # check all zeros
                for z in zeros:
                    t = tol or \
                        1000 * max(float_info.epsilon, np.abs(z) * sqrt_eps)
                    idx = np.where(np.abs(z - poles) < t)[0]
                    if len(idx):
                        # cancel this zero against THE CLOSEST of the poles
                        poles = np.delete(poles, np.argmin(np.abs(z - poles)))
                    else:
                        # keep this zero
                        newzeros.append(z)


                # poly([]) returns a scalar, but we always want a 1d array
                num[i][j] = np.atleast_1d(gain * np.real(np.poly(newzeros)))
                den[i][j] = np.atleast_1d(np.real(np.poly(poles)))
                if debug:
                    print('\n\n\n\nEND')
                    polesOld = np.roots(TF.den[i][j])
                    import code
                    code.interact(local=locals())
                    # TODO: Need to ensure that pole-zero cancellations of complex conjugates cancel each other, and not a real zero cancelling an imaginary pole.

        # end result
        return c.TransferFunction(num, den, TF.dt)

# IDK what is up with the python control MIMO multiplication, but it does not seem to be correct
# So here is a fix for without using the underlying __mul__ method, which seems to cause some issues.
def multiply(A, B, sign = 1):
    if A.noutputs != B.ninputs:
        raise ValueError(f'Dimension mismatch between A (n={A.noutputs}) and B (n={B.ninputs})')
    nums = []
    dens = []
    for i in range(A.noutputs):
        nums_o = []
        dens_o = []
        for j in range(B.ninputs):
            G = c.tf('s')*0
            for k in range(A.noutputs):
                G += sign*A[i, k]*B[k, j]
            nums_o.append(G.num[0][0])
            dens_o.append(G.den[0][0])
        nums.append(nums_o)
        dens.append(dens_o)
    return c.TransferFunction(nums, dens, dt = A.dt)


def transpose(A):
    nums = []
    dens = []
    for i in range(A.ninputs):
        nums.append([A.num[k][i] for k in range(A.ninputs)])
        dens.append([A.den[k][i] for k in range(A.ninputs)])
    return c.TransferFunction(nums, dens, dt = A.dt)


def concatenateTFs(tfs):
    nums = []
    dens = []
    for row in tfs:
        _nums = []
        _dens = []
        for _tf in row:
            dt = _tf.dt
            _nums.append(_tf.num[0][0])
            _dens.append(_tf.den[0][0])
        nums.append(_nums)
        dens.append(_dens)
    return c.TransferFunction(nums, dens, dt = dt)



def addScalar_MIMO(x, A, diagonalOnly = False):
    nums = []
    dens = []
    for i in range(A.noutputs):
        nums_o = []
        dens_o = []
        for j in range(A.ninputs):
            if diagonalOnly:
                if i == j:
                    G = A[i, j] + x
                else:
                    G = A[i, j]
            else:
                G = A[i, j] + x
            nums_o.append(G.num[0][0])
            dens_o.append(G.den[0][0])
        nums.append(nums_o)
        dens.append(dens_o)
    return c.TransferFunction(nums, dens, dt = A.dt)


def elementWise_sensitivity(L):
    nums = []
    dens = []
    for _o in range(L.noutputs):
        _nums = []
        _dens = []
        for _i in range(L.ninputs):
            _tf = L[_o, _i]
            dt = _tf.dt
            _S = sensitivity(_tf)
            _nums.append(_S.num[0][0])
            _dens.append(_S.den[0][0])
        nums.append(_nums)
        dens.append(_dens)
    return c.TransferFunction(nums, dens, dt = dt)


def sensitivity(L, method = 'laplace_domain'):
    '''
    Compute the sensitivity functon of a (system) of transfer function(s)
    '''
    if L.issiso():
        if method.lower() == 'laplace_domain':
            return 1/(1+c.tf(L))
        elif method.lower() == 'time_domain':
            A, B, C, D = L.A, L.B, L.C, L.D
            D = 1 + D
            if np.isfinite(np.linalg.cond(D)):
                A_inv = A - B @ np.linalg.inv(D) @ C
                B_inv = B @ np.linalg.inv(D)
                C_inv = -np.linalg.inv(D) @ C
                D_inv = np.linalg.inv(D)
                sys_inv = c.ss(A_inv, B_inv, C_inv, D_inv)
            return sys_inv
    else:
        if method.lower() == 'laplace_domain':
            if isinstance(L, c.iosys.LinearIOSystem):
                L = c.ss2tf(L)
            L1 = addScalar_MIMO(1, L, diagonalOnly = True)
            return invert_MIMO(L1, method = 'laplace_domain')
        elif method.lower() == 'time_domain':
            if isinstance(L, c.xferfcn.TransferFunction):
                L = c.tf2ss(L)
            A, B, C, D = L.A, L.B, L.C, L.D
            D1 = np.eye(D.shape[0]) + D
            L1 = c.ss(A, B, C, D1)
            return invert_MIMO(L1, method = 'time_domain')


def cofactor(matrix, i, j):
    '''
    Get the cofactor of a matrix for row i and column j
    '''
    Mij = np.delete(np.delete(matrix, i, axis = 0), j, axis = 1)
    return (-1)**(i+j) * determinant(Mij)


def determinant(matrix):
    '''
    Find the determinant of a matrix by recursively solving for the determinants of
    the sub-matrices until we hit the base, 2x2, case. 
    '''
    # Get size of the matrix
    N = len(matrix)
    # Check if base case (2x2 matrix)
    if N == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
    # Start computation of determinant
    det = 0
    for j in range(N):
        Cfctr = cofactor(matrix, 0, j)
        det += matrix[0][j] * Cfctr
    return det


def invert_MIMO(G, method = 'time_domain'):
    '''
    Get the inverse of a MIMO transfer function
    '''
    if method.lower() == 'laplace_domain':
        # Check if G is a transfer function
        return_type = 'TransferFunction'
        if isinstance(G, c.xferfcn.TransferFunction):
            # Convert to list of list of transfections
            G = _TF2Lists(G)
        elif isinstance(G, c.iosys.LinearIOSystem):
            G = _TF2Lists(c.tf(G))
            return_type = 'StateSpace'
        
        N = len(G)

        # Determinant of G, if zero, raise error
        detG = determinant(G)
        if detG == 0:
            raise ValueError('Inversion failed: Singular matrix.')
        detG = minreal(detG)
        if N > 2:
            # Using 1/|G| * adjoint(G) => 1/|G| * cofactor(G)^T
            # Adjoint of G
            adjG = [[cofactor(G, j, i) for j in range(N)] for i in range(N)]

            # # Get inverse
            # invG = [[minreal(adjG[i][j] / detG) for j in range(N)] for i in range(N)]
            # Get inverse
            invG = [[minreal(adjG[i][j]) / detG for j in range(N)] for i in range(N)]
        else:
            invG = [[G[1][1]/detG, -G[0][1]/detG], [-G[1][0]/detG, G[0][0]/detG]]

        if return_type.lower() == 'transferfunction':
            return _Lists2TF(invG)
        else:
            return c.ss2tf(_Lists2TF(invG))
    
    elif method.lower() == 'time_domain':
        # Check if G is a transfer function
        return_type = 'StateSpace'
        if isinstance(G, c.xferfcn.TransferFunction):
            # Convert to list of list of transfections
            return_type = 'TransferFunction'
            G = c.tf2ss(G)
        elif isinstance(G, list):
            G = c.tf2ss(_Lists2TF(G))
        
        A, B, C, D = G.A, G.B, G.C, G.D
        if np.isfinite(np.linalg.cond(D)):
            A_inv = A - B @ np.linalg.inv(D) @ C
            B_inv = B @ np.linalg.inv(D)
            C_inv = -np.linalg.inv(D) @ C
            D_inv = np.linalg.inv(D)
            sys_inv = c.ss(A_inv, B_inv, C_inv, D_inv)
            if return_type.lower() == 'transferfunction':
                return c.ss2tf(sys_inv)
            else:
                return sys_inv
        else:
            raise ValueError('Cannot do time-domain based inversion with sinuglar D matrix.') 

    else:
        raise ValueError(f'Unknown method for MIMO inverse: {method}')


def _TF2Lists(TF):
    ''' 
    Convert a transfer function object into a list (noutputs) of list (ninputs) of transfer functions
    '''
    GList = []
    for _o in range(TF.noutputs):
        rows = []
        for _i in range(TF.ninputs):
            rows.append(TF[_o, _i])
        GList.append(rows)
    return GList


def _Lists2TF(G):
    '''
    Convert a list of list of transfer functions into a single transfer function object
    '''
    return concatenateTFs(G)


def getAllOpenLoopTransfer_ss(K, G, at = 'output'):
    '''
    Get loop transfer function at plant "input" or plant "output"
    '''
    if not isinstance(G, c.iosys.LinearIOSystem) or not isinstance(K, c.iosys.LinearIOSystem):
        raise ValueError('K and G need to be state space systems.')
    
    if at.lower() == 'output':
        AG, BG, CG, DG = G.A, G.B, G.C, G.D
        AK, BK, CK, DK = K.A, K.B, K.C, K.D
        if np.all(np.isclose(np.hstack((AK, BK)), 0)) and np.all(np.isclose(CK, 0)):
            # We have direct feedthrough, assume controller states are same as model, so
            A_ol = AG.copy()
            B_ol = BG @ DK
            C_ol = CG.copy()
            D_ol = DG @ DK
        else:
            A_ol = np.block([
                [AG, BG @ CK],
                [np.zeros((AK.shape[0], AG.shape[1])), AK]
            ])
            B_ol = np.block([
                [BG @ DK],
                [BK]
            ])
            C_ol = np.block([
                [CG, DG @ CK]
            ])
            D_ol = np.block([
                [DG @ DK]
            ])
        ss_ol = c.ss(A_ol, B_ol, C_ol, D_ol)
        return ss_ol

    elif at.lower() == 'input':
        AG, BG, CG, DG = G.A, G.B, G.C, G.D
        AK, BK, CK, DK = K.A, K.B, K.C, K.D

        if np.all(np.isclose(np.hstack((AK, BK)), 0)) and np.all(np.isclose(CK, 0)):
            # We have direct feedthrough, assume controller states are same as model, so
            A_ol = AG.copy()
            B_ol = BG.copy()
            C_ol = DK @ CG
            D_ol = DK @ DG
        else:
            A_ol = np.block([
                [AG, np.zeros((AG.shape[0], AK.shape[1]))],
                [BK @ CG, AK]
            ])
            B_ol = np.block([
                [BG],
                [BK @ DG]
            ])
            C_ol = np.block([
                [DK @ CG, CK]
            ])
            D_ol = np.block([
                [DK @ DG]
            ])
        ss_ol = c.ss(A_ol, B_ol, C_ol, D_ol)
        return ss_ol
    else:
        raise ValueError('Please specify where to take the loop transfer: at = "input" of the plant or at = "output" of the plant')


def diskmargin_siso(L, skew = 0, omega = np.logspace(-4, 3, 5000), res = 2000, plot = False, returnMarginOnly = False, color = 'tab:blue'):
    results = {}

    # Get sensitivity function
    S = sensitivity(L, method='laplace_domain')
    # Derive peak magnitude of S, and associated frequency
    Smax, omegaMax, idxMax, mag, phase = _peakSensitivity(S + (skew - 1)/2, omega = omega)

    # Get disk margin
    alpha_max = 1/Smax

    # Get limit value
    cond = 2/np.abs(1 + skew)

    # Case: Alpha_max > 2 -> Disk exterior
    if alpha_max > cond:
        # Get margins
        [gamma_min, gamma_max], PM_max, radius, center = _getDiskParams(alpha_max, skew)
        GM = np.nanmin([1/gamma_min, gamma_max])

        _gamma_max = -gamma_min
        _gamma_min = -gamma_max
        GM = np.inf
        # Derive simultaneous gain and phase perturbation margins
        if np.abs(_gamma_max - _gamma_min) > 100:
            GMContour = np.logspace(np.log10(_gamma_min), np.log10(_gamma_max), res)
        else:
            GMContour = np.linspace(_gamma_min, _gamma_max, res)
        PMContour = _thetaFromGamma(-1*GMContour, gamma_min, gamma_max)

        # Find destabilizing perturbation f0
        delta0 = 1/(S(omegaMax*1j) + (skew-1)/2)
        f0 = (2 + (1-skew)*delta0)/(2 - (1+skew)*delta0)
        GM_worst = -1*np.abs(f0)
        PM_worst = 180/np.pi * np.arctan(f0.imag/f0.real)

        # Gain and phase margins as function of frequency
        [gamma_mins, gamma_maxs], PMs, radii, centrum = _getDiskParams(1/mag, skew)
        GMs = np.nanmin(np.vstack((1/gamma_mins, gamma_maxs)), axis = 0)



    # Case: Alpha_max = 2 -> Half-plane
    # -> PM=inf, GM=inf -> there is no upper bound, only lower, and there is no 
    elif alpha_max == cond:
        # Get margins
        gamma_min = (2 - alpha_max*(1-skew))/(2 + alpha_max*(1+skew))
        gamma_max = np.inf
        radius = np.nan
        center = np.nan
        PM_max = np.inf
        GM = np.inf if np.isclose(gamma_min, 0) else gamma_min
        # Derive simultaneous gain and phase perturbation margins
        GMContour = [gamma_min, np.inf]
        PMContour = [-np.inf, np.inf]

        # Find destabilizing perturbation f0
        delta0 = 1/(S(omegaMax*1j) + (skew-1)/2)
        if delta0 != 2:
            f0 = (2 + (1-skew)*delta0)/(2 - (1+skew)*delta0)
            GM_worst = np.abs(f0)
            PM_worst = 180/np.pi * np.arctan(f0.imag/f0.real)
        else:
            f0 = np.nan
            GM_worst = np.nan
            PM_worst = np.nan

        # Gain and phase margins as function of frequency
        PMs = np.array([np.inf, np.inf])
        GMs = np.array([gamma_min, GM])

    

    # Case: Alpha_max < 2 -> Disk interior
    else:
        # Get margins
        [gamma_min, gamma_max], PM_max, radius, center = _getDiskParams(alpha_max, skew)
        GM = np.nanmin([1/gamma_min, gamma_max])

        _gamma_max = gamma_max
        _gamma_min = gamma_min
        # Derive simultaneous gain and phase perturbation margins
        if np.abs(_gamma_max - _gamma_min) > 100:
            GMContour = np.logspace(np.log10(_gamma_min), np.log10(_gamma_max), res)
        else:
            GMContour = np.linspace(_gamma_min, _gamma_max, res)
        PMContour = _thetaFromGamma(GMContour, gamma_min, gamma_max)
        
        # Find destabilizing perturbation f0
        delta0 = 1/(S(omegaMax*1j) + (skew-1)/2)
        f0 = (2 + (1-skew)*delta0)/(2 - (1+skew)*delta0)
        GM_worst = np.abs(f0)
        PM_worst = 180/np.pi * np.arctan(f0.imag/f0.real)

        # Gain and phase margins as function of frequency
        [gamma_mins, gamma_maxs], PMs, radii, centrum = _getDiskParams(1/mag, skew)
        GMs = np.nanmin(np.vstack((1/gamma_mins, gamma_maxs)), axis = 0)



    if plot:
        fig = plt.figure(figsize = (7.2, 5.4))
        ax = fig.add_subplot(111)
        vals = r' $\alpha_{max}$ ' + '={:.2f}, '.format(alpha_max) + r'$PM_{max}$' + '={:.2f} deg, '.format(PM_max*180/np.pi) + r'$GM_{max}=\pm$' + '{:.2f} dB'.format(np.max(c.mag2db(GMContour)).real)
        if alpha_max > 2:
            _plotDiskMarginExclusion(GMContour, PMContour, parentAx=ax, color = color, title = vals)
            ax.scatter(GM_worst, np.abs(PM_worst), color = color)
        elif alpha_max == 2:
            _plotDiskMarginHalfPlane(GMContour, PMContour, parentAx=ax, color = color, title = vals)
        else:
            _plotDiskMarginVariation(GMContour, PMContour, parentAx=ax, color = color, title = vals)
            ax.scatter(c.mag2db(GM_worst), np.abs(PM_worst), color = color)
        handles, _ = ax.get_legend_handles_labels()
        plotting.addLegendLine(handles, linestyle = 'none', marker = 'o', label = 'Destabilizing\nperturbation', color = color)
        plotting.addLegendPatch(handles, color = color, alpha = 0.2, label = 'Stable region')
        ax.legend(handles=handles)
        plotting.prettifyAxis(ax)
        plt.tight_layout()

        if alpha_max != 2:
            fig2 = _plotGMPM(GMs, PMs, omega)
        else:
            fig2 = _plotGMPM(GMs, PMs, [omega[0], omega[-1]])

    if returnMarginOnly:
        # Returns disk margin, maximum and minimum gain factors (absolute), maximum phase margin (deg), worst perturbation
        return alpha_max, [gamma_min, gamma_max], GM, PM_max*180/np.pi, f0
    else:
        # Package results
        results.update({'DM':alpha_max})
        results.update({'GM':[gamma_min, gamma_max]})
        results.update({'PM':PM_max*180/np.pi})
        results.update({'omega_max':omegaMax})
        results.update({'sensitivity':S})
        results.update({'gammas':GMContour})
        results.update({'thetas':PMContour})
        results.update({'delta0':delta0})
        results.update({'f0':f0})
        results.update({'GM_worst':GM_worst})
        results.update({'PM_worst':PM_worst})
        results.update({'Smax':Smax})
        results.update({'Sperturb':sensitivity(f0*L)})
        results.update({'disk_radius':radius})
        results.update({'disk_center':center})
        results.update({'GM(s)':GMs})
        results.update({'PM(s)':PMs})
        if plot:
            results.update({'figures':[fig, fig2]})
        else:
            results.update({'figures':None})
        return results



def _peakSensitivity(S, omega = np.logspace(-4, 3, 5000)):
    # Derive peak magnitude of S, and associated frequency
    mag, phase, _ = c.bode(S, omega = omega, plot = False)
    idxMax = np.argmax(mag)
    omegaMax = omega[idxMax]
    Smax = mag[idxMax]
    return Smax, omegaMax, idxMax, mag, phase


def _getDiskParams(alpha, skew):
    # Obtain maximum gain (only) variations
    gamma_min = (2 - alpha*(1-skew))/(2 + alpha*(1+skew))
    gamma_max = (2 + alpha*(1-skew))/(2 - alpha*(1+skew))
    # Define complex plane disk properties
    radius = (1/2)*np.abs(gamma_max - gamma_min)
    center = (1/2)*(gamma_max + gamma_min)
    # Get maximum phase (only) variation
    PM_max = _thetaFromGamma(1, gamma_min, gamma_max)
    return [gamma_min, gamma_max], PM_max, radius, center


def _plotDiskMarginExclusion(GMContour, PMContour, parentAx = None, color = 'tab:blue', title = ''):
    if parentAx is None:
        fig = plt.figure(figsize = (7.2, 5.4))
        ax = fig.add_subplot(111)
    else:
        fig = None
        ax = parentAx
    ax.set_title(r'$\mathbf{Disk}$ $\mathbf{margins}$:' + title)
    ax.plot(-1*GMContour, PMContour*180/np.pi, color = color)
    ax.scatter(1, 0, marker = '+', s = 100, linewidth = 5, color = 'firebrick', label = 'Nominal system')
    xlim = ax.get_xlim()
    x = np.hstack([3*np.nanmin(-1*GMContour), -1*GMContour, 100])
    ylower = np.hstack([0, PMContour*180/np.pi, 0])
    yupper = np.ones(len(x))*10*np.nanmax(PMContour)*180/np.pi
    ax.fill_between(x, yupper, ylower, color = color, alpha = 0.2)
    ax.set_ylim([0, 1.1*np.nanmax(PMContour)*180/np.pi])
    ax.set_xlim([xlim[0], 20])
    ax.grid('on')
    ax.set_ylabel(r'$\mathbf{Phase}$ $\mathbf{variation}$, deg')
    ax.set_xlabel(r'Multiplicative $\mathbf{Gain}$ $\mathbf{variation}$, -')
    plotting.prettifyAxis(ax)
    ax.legend()
    if fig is not None:
        return fig
    
def _plotDiskMarginVariation(GMContour, PMContour, parentAx = None, color = 'tab:blue', title = ''):
    if parentAx is None:
        fig = plt.figure(figsize = (7.2, 5.4))
        ax = fig.add_subplot(111)
    else:
        fig = None
        ax = parentAx
    ax.set_title(r'$\mathbf{Disk}$ $\mathbf{margins}$:' + title)
    ax.plot(c.mag2db(GMContour), PMContour*180/np.pi, color = color)
    ax.fill_betweenx(PMContour*180/np.pi, c.mag2db(GMContour), 0, color = color, alpha = 0.2)
    ylims = ax.get_ylim()
    ax.set_ylim([0, ylims[1]])
    ax.grid('on')
    ax.set_ylabel(r'$\mathbf{Phase}$ $\mathbf{variation}$, deg')
    ax.set_xlabel(r'$\mathbf{Gain}$ $\mathbf{variation}$, dB')
    plotting.prettifyAxis(ax)
    if fig is not None:
        return fig


def _plotDiskMarginHalfPlane(GMContour, PMContour, parentAx = None, color = 'tab:blue', title=''):
    if parentAx is None:
        fig = plt.figure(figsize = (7.2, 5.4))
        ax = fig.add_subplot(111)
    else:
        fig = None
        ax = parentAx
    ax.set_title(r'$\mathbf{Disk}$ $\mathbf{margins}$:' + title)
    ax.set_ylim([0, 90])
    if GMContour[0] > 0:
        ax.scatter(c.mag2db(GMContour[0]), 0, alpha = 0)
        xlims = ax.get_xlim()
        ax.axvspan(c.mag2db(GMContour[0]), 100, color = color, alpha = 0.2)
        ax.axvline(c.mag2db(GMContour[0]), -180, 180)
    else:
        xlims = [-10, 10]
        ax.axvspan(0, 100, color = color, alpha = 0.2)
        ax.axvline(c.mag2db(GMContour[0]), -180, 180)
    ax.set_xlim(xlims)
    ax.grid('on')
    ax.set_ylabel(r'$\mathbf{Phase}$ $\mathbf{variation}$, deg')
    ax.set_xlabel(r'$\mathbf{Gain}$ $\mathbf{variation}$, dB')
    plotting.prettifyAxis(ax)
    if fig is not None:
        return fig



def _plotGMPM(GMs, PMs, omega, axes = [None, None], color = 'tab:blue'):
    if axes[0] is None or axes[1] is None:
        fig = plt.figure(figsize = (10, 6))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex = ax1)
    else:
        fig = None
        ax1 = axes[0]
        ax2 = axes[1]
    ax1.semilogx(omega, c.mag2db(GMs), color = color, linewidth = 2)
    ax2.semilogx(omega, PMs*180/np.pi, color = color, linewidth = 2)
    ax1.set_ylabel(r'$\mathbf{Gain}$ $\mathbf{margin}$, dB')
    ax2.set_ylabel(r'$\mathbf{Phase}$ $\mathbf{margin}$, deg')
    ax2.set_xlabel(r'$\mathbf{Frequency}$, rad/s')
    ax1.grid('on')
    ax2.grid('on')
    # plotting.prettifyAxis(ax1)
    # plotting.prettifyAxis(ax2)
    plt.tight_layout()
    if fig is not None:
        return fig
    


def _thetaFromGamma(gamma, gamma_min, gamma_max):
    num = gamma**2 + gamma_min*gamma_max
    den = gamma*(gamma_min + gamma_max)
    ratio = np.around(num/den, 6)
    idxsInf = np.where(np.abs(ratio) > 1)[0]
    if len(idxsInf):
        ratio[idxsInf] = 0
    theta = np.arccos(ratio)
    if len(idxsInf):
        theta[idxsInf] = np.inf
    return theta



def getAlpha_PMdes(PMdes, gamma = 1, skew = 0):
    # Function to get alpha from desired phase margin
    num = 4*gamma**2 + 4 - 8*gamma*np.cos(PMdes)
    den = 2*gamma*np.cos(PMdes)*(1-skew)*(1+skew) + gamma**2 * (1+skew)**2 + (1-skew)**2
    alpha = np.sqrt(num/den)
    return alpha





# def getLoopTransferFunction_at_ij(K, G, i, j, at = 'output'):
#     '''
#     Get loop transfer function at plant "input" or plant "output"
#     # Assumes positive feedback
#     '''
#     if not isinstance(G, c.iosys.LinearIOSystem) or not isinstance(K, c.iosys.LinearIOSystem):
#         raise ValueError('K and G need to be state space systems.')
    
#     if i != j:
#         raise NotImplementedError('Currently, loop transfers for off-diagonals is not verified or implemented completely.')
    
#     if at.lower() == 'output':
#         AG, BG, CG, DG = G.A, G.B, G.C, G.D
#         AK, BK, CK, DK = K.A, K.B, K.C, K.D

#         # CG = -1*CG
#         # DG = -1*DG

#         DgDk = DG @ DK
#         DgDk_spliced = np.copy(DgDk)
#         DgDk_spliced[:, i] = 0
#         DK_correction = np.linalg.inv(np.eye(len(DG)) - DgDk_spliced)
#         cDgDki = DK_correction @ DgDk_spliced[:, i].reshape(-1, 1)

#         cCg = DK_correction @ CG
#         cDgCk = DK_correction @ DG @ CK

#         BgDk = BG @ DK
#         BgDki = BgDk[:, i].reshape(-1, 1) 
#         BgDk_spliced = np.copy(BgDk)
#         BgDk_spliced[:, i] = 0
#         Bki = BK[:, i].reshape(-1, 1)
#         Bk_spliced = np.copy(BK)
#         Bk_spliced[:, i] = 0


#         # A_cl = np.block([
#         #     [AG + BgDk_spliced @ cCg, BG @ CK + BgDk_spliced @ cDgCk],
#         #     [Bk_spliced @ cCg, AK + Bk_spliced @ cDgCk]
#         # ])
#         # B_cl = np.block([
#         #     [BgDki],
#         #     [Bki]
#         # ])
#         # C_cl = np.block([
#         #     [cCg, cDgCk]
#         # ])
#         # D_cl = np.block([
#         #     [cDgDki]
#         # ])


#         A_cl = np.block([
#             [AG + BgDk_spliced @ cCg, BG @ CK + BgDk_spliced @ cDgCk],
#             [Bk_spliced @ cCg, AK + Bk_spliced @ cDgCk]
#         ])
#         B_cl = np.block([
#             [BgDki + BgDk_spliced @ cDgDki],
#             [Bki + Bk_spliced @ cDgDki]
#         ])
#         C_cl = -1*np.block([
#             [cCg, cDgCk]
#         ])
#         D_cl = -1*np.block([
#             [cDgDki]
#         ])        

#         # A_cl = np.block([
#         #     [AG, BG @ CK],
#         #     [np.zeros((AK.shape[0], AG.shape[1])), AK]
#         # ])
#         # B_cl = np.block([
#         #     [BgDki],
#         #     [Bki]
#         # ])
#         # C_cl = np.block([
#         #     [cCg, cDgCk]
#         # ])
#         # D_cl = np.block([
#         #     [cDgDki]
#         # ])

#         # A_cl = np.block([
#         #     [AG + BgDk_spliced @ cCg, BG @ CK + BgDk_spliced @ cDgCk],
#         #     [Bk_spliced @ cCg, AK + Bk_spliced @ cDgCk]
#         # ])
#         # B_cl = np.block([
#         #     [BgDki],
#         #     [Bki]
#         # ])
#         # C_cl = np.block([
#         #     [cCg, cDgCk]
#         # ])
#         # D_cl = np.block([
#         #     [cDgDki]
#         # ])        

#         ss_cl = c.ss(A_cl, B_cl, C_cl, D_cl)

#         # import code
#         # code.interact(local=locals())

#         return ss_cl[j, 0]


#     elif at.lower() == 'input':
#         AG, BG, CG, DG = G.A, G.B, G.C, G.D
#         AK, BK, CK, DK = K.A, K.B, K.C, K.D
        
#         DG_spliced = np.copy(DG)
#         DG_spliced[:, j] = 0
#         DG_correction = np.linalg.inv(np.eye(len(DK)) - DG_spliced)
#         DGj = DG_correction @ DG[:, j].reshape(-1, 1)

#         DgcDkCg = DG_correction @ DK @ CG   # x
#         DgcCk = DG_correction @ CK          # xk

#         B = np.vstack([BG, BK @ DG])
        
#         A11_cl = AG + BG @ DgcDkCg
#         A12_cl = BG @ DgcCk
#         A21_cl = (BK @ CG) + (BK @ DG) @ DgcDkCg
#         A22_cl = AK + (BK @ DG) @ DgcCk
#         A_cl = np.block([
#             [A11_cl, A12_cl],
#             [A21_cl, A22_cl]
#             ])
#         B_cl = B @ DGj
#         C_cl = np.block([
#             [DgcDkCg, DgcCk]
#         ])
#         D_cl = DGj

#         ss = c.ss(A_cl, B_cl, C_cl, D_cl)
#         return ss[i, 0]

#     else:
#         raise ValueError('Please specify where to take the loop transfer: at = "input" of the plant or at = "output" of the plant')

