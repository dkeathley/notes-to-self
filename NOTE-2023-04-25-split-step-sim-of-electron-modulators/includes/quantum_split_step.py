import numpy as np

def quantum_split_step(psi0, V, t, z, v0=0):
    """
    This function carries out the split step procedure 
    to calculate a wavefunction within some arbitrary potential V(t, z).  
    The potential region is a real space function of both length (z) 
    and time (t).  
    
    psill units are atomic units.
    
    INPUTS: 
    
        psi0:   input wavefunction (as a function of space)
        t:      time vector
        z:      space vector (moving frame)
        V:      potential matrix (function of space and time)
        v0:     Optional central momentum value (for moving frame)
 
    OUTPUTS:
        psi:    output wavefunction (as function of space for each point in time)
    """

    #Number of time steps to go through
    Nt = t.size
    
    #z should be uniformly sampled
    dt = t[1]-t[0]
    dz = z[1]-z[0]

    # initialize 
    psi = np.zeros( (Nt, psi0.size)) + 0j
    psi_f = np.zeros( (Nt, psi0.size) ) + 0j
    psi[0, :] = psi0
    psi_f[0, :] = dt*np.fft.fft(psi0);
    
    K = np.fft.fftfreq(psi0.size, d=dz)

    #calculate all the traces and spectra
    for co in range(1,Nt):
        
        #spectrum calculation
        psitemp_f = dz * np.exp(-1j*K**2*dt/4)*\
        np.fft.fft( np.exp(-1j*V(t[co], z + t[co]*v0)*dt)*\
        np.fft.ifft( np.exp(-1j*K**2*dt/4)*\
        np.fft.fft( psi[co-1, :] ) ) )*np.exp(1j*v0**2/2*dt);

        #assignments
        psi_f[co, :] = psitemp_f;
        psi[co, :] = 1/dz*np.fft.ifft(psitemp_f);
    
    
    return psi

def quantum_split_step2(psi0, V, t, z, v0=0):
    """
    This function carries out the split step procedure 
    to calculate a wavefunction within some arbitrary potential V(t, z).  
    The potential region is a real space function of both length (z) 
    and time (t).  
    
    psill units are atomic units.
    
    INPUTS: 
    
        psi0:   input wavefunction (as a function of space)
        t:      time vector
        z:      space vector (moving frame)
        V:      potential matrix (function of space and time)
        v0:     Optional central momentum value (for moving frame)
 
    OUTPUTS:
        psi:    output wavefunction (as function of space for each point in time)
    """

    #Number of time steps to go through
    Nt = t.size
    
    #z should be uniformly sampled
    dt = t[1]-t[0]
    dz = z[1]-z[0]

    # initialize 
    psi = np.zeros( (Nt, psi0.size)) + 0j
    psi_f = np.zeros( (Nt, psi0.size) ) + 0j
    psi[0, :] = psi0
    psi_f[0, :] = dt*np.fft.fft(psi0);
    
    K = np.fft.fftfreq(psi0.size, d=dz)

    #calculate all the traces and spectra
    for co in range(1,Nt):
        
        #spectrum calculation
        psitemp = np.exp(-1j*V(t[co], z + t[co]*v0)*dt/2)*\
        np.fft.ifft(np.exp(-1j*K**2*dt/2)*\
        np.fft.fft( np.exp(-1j*V(t[co], z + t[co]*v0)*dt/2)*psi[co-1, :]))*\
        np.exp(1j*v0**2/2*dt)

        #assignments
        psi[co, :] = psitemp;    
    
    return psi