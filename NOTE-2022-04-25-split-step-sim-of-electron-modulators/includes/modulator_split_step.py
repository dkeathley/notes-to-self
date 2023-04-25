def modulator_split_step(A0, z, f, dt, beta2, beta3, gamma):
    """
    This function carries out the split step procedure 
    to calculate a pulse spectra propagating through the NLSE. The
    split-step implementation here is not optimized in any way. (I'd say
    that it's relatively inefficient.) This function does not use a
    normalized NLSE. (4-21-20).
    
    INPUTS: 
        psi0:   input wavefunction (as a function of space)
        t:      time vector
        z:      space vector (moving frame)
 
    OUTPUTS:
        psi     output wavefunction (as function of space for each point in time)
    """

    Nt = t.size
    
    #z should be uniformly sampled
    dt = t[1]-t[0]

    # initialize 
    A = np.zeros( (Nz, A0.size)) + 0j
    A_f = np.zeros( (Nz, A0.size) ) + 0j
    A[0, :] = A0
    A_f[0, :] = dt*np.fft.fft(A0);

    #calculate all the traces and spectra
    for co in range(1,Nz):
        #spectrum calculation
        Atemp_f = dt * np.exp(-1j/4*beta2*(2*np.pi*f)**2*dz + 1j/12*beta3*(2*np.pi*f)**3*dz)*\
        np.fft.fft( np.exp(-1j*gamma*np.abs(A[co-1, :])**2*dz)*\
        np.fft.ifft( np.exp(-1j/4*beta2*(2*np.pi*f)**2*dz + 1j/12*beta3*(2*np.pi*f)**3*dz)*\
        np.fft.fft( A[co-1, :] ) ) );

        #assignments
        A_f[co, :] = Atemp_f;
        A[co, :] = 1/dt*np.fft.ifft(Atemp_f);
    
    
    return psi