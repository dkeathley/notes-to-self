def quantum_split_step(psi0, V, t, z):
    """
    This function carries out the split step procedure 
    to calculate a wavefunction within some arbitrary potential V(t, z).  
    The potential region is a real space function of both length (z) 
    and time (t).  
    
    All units are atomic units.
    
    INPUTS: 
    
        psi0:   input wavefunction (as a function of space)
        t:      time vector
        z:      space vector (moving frame)
        V:      potential matrix (function of space and time)
 
    OUTPUTS:
        psi:    output wavefunction (as function of space for each point in time)
    """

    #Number of time steps to go through
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