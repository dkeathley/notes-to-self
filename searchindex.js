Search.setIndex({"docnames": ["NOTE-2022-03-30-short-cable-impedance-tline-vs-lumped-element/ipynb/note", "NOTE-2022-04-04-hemoglobin-refractive-index-vis-nir/ipynb/note", "NOTE-2022-04-12-surface-plasmon-polariton-mode-dispersion/ipynb/note", "NOTE-2022-11-27-coupled-oscillators/ipynb/note-resonances-in-coupled-nanoantennas", "intro"], "filenames": ["NOTE-2022-03-30-short-cable-impedance-tline-vs-lumped-element/ipynb/note.ipynb", "NOTE-2022-04-04-hemoglobin-refractive-index-vis-nir/ipynb/note.ipynb", "NOTE-2022-04-12-surface-plasmon-polariton-mode-dispersion/ipynb/note.ipynb", "NOTE-2022-11-27-coupled-oscillators/ipynb/note-resonances-in-coupled-nanoantennas.ipynb", "intro.md"], "titles": ["Short Cable Impedance \u2013 T-Line vs. Lumped Element Approximation", "Kramers Kronig Analysis of Hemoglobin", "Surface Plasmon Polariton Mode Dispersion with Derivation", "Electrically-Coupled Nanoantenna Resonators", "Technical Notes to Self"], "terms": {"thi": [0, 1, 2, 3], "note": [0, 1, 2], "examin": [0, 3], "two": [0, 1, 3], "approach": [0, 1, 2, 3], "coax": 0, "The": [0, 1, 3], "transmiss": 0, "It": [0, 1], "show": [0, 1], "both": [0, 1, 3], "ar": [0, 1, 3, 4], "same": [0, 1, 3], "limit": 0, "howev": [0, 1, 3], "more": [0, 3], "gener": [0, 1, 2, 3], "sinc": [0, 3], "just": [0, 1, 3], "easi": 0, "straightforward": 0, "i": [0, 1, 2, 3, 4], "don": 0, "see": [0, 1, 3], "why": 0, "you": [0, 1, 3], "shouldn": 0, "alwai": 0, "calcul": 0, "us": [0, 3, 4], "pdf": [0, 4], "wa": [1, 3], "develop": [1, 3], "want": 1, "find": [1, 2, 3], "method": 1, "biolog": 1, "particular": [1, 3], "simul": [1, 3], "hyperspectr": 1, "optic": [1, 3], "field": [1, 3], "resolv": 1, "microscop": 1, "Then": [1, 3], "focu": [1, 2], "particularli": 1, "can": [1, 3], "appli": [1, 3], "host": 1, "where": [1, 3], "spectra": [1, 3], "known": [1, 3], "In": [1, 2, 3], "1": [1, 3], "full": [1, 3, 4], "model": 1, "includ": [1, 3], "shape": [1, 3], "describ": 1, "There": [1, 3], "thei": [1, 3], "address": 1, "complex": 1, "purpos": [1, 3], "suggest": 1, "2": [1, 3], "how": [1, 3], "determin": [1, 3], "complet": 1, "measur": 1, "form": 1, "unoxygen": 1, "Their": 1, "base": [1, 3], "dataset": 1, "provid": [1, 3], "scott": 1, "prahl": 1, "found": 1, "through": [1, 3], "websit": 1, "perform": [1, 3], "follow": [1, 2], "take": [1, 3], "experiment": [1, 3], "absorpt": 1, "directli": [1, 3], "mu_": 1, "deriv": [1, 4], "here": [1, 3], "kappa": 1, "omega": [1, 3], "eq": 1, "n": [1, 3], "real": [1, 3], "part": 1, "we": [1, 3], "three": [1, 3], "step": 1, "comput": 1, "so": [1, 3], "inform": 1, "incorpor": 1, "fdtd": [1, 3], "electromagnet": [1, 2, 3], "also": [1, 3], "dispers": [1, 4], "block": 1, "load": [1, 3], "ani": [1, 3], "need": [1, 3], "packag": 1, "defin": [1, 3], "core": 1, "remaind": 1, "n_kk": 1, "y": [1, 3], "y_0": 1, "n_0": 1, "dw_prime": 1, "wavelength": [1, 3], "along": [1, 2], "singl": [1, 3], "fix": 1, "medium": 1, "eps_drude_lorentz": 1, "p": [1, 3], "coeffici": 1, "vector": 1, "those": [1, 3], "epsilon": 1, "residu": 1, "eps_mea": 1, "conveni": [1, 3], "scipi": 1, "optim": 1, "least_squar": 1, "permitt": [1, 3], "e": [1, 3], "g": [1, 3], "tool": 1, "like": [1, 3], "setup": 1, "workspac": 1, "reset": 1, "import": [1, 3], "numpi": [1, 3], "np": [1, 3], "matplotlib": [1, 3], "pyplot": [1, 3], "plt": [1, 3], "io": 1, "sio": 1, "interpol": 1, "sco": 1, "sy": [1, 3], "path": [1, 3], "append": [1, 3], "physical_constants_si": 1, "pcsi": 1, "def": 1, "evalu": 1, "frequenc": [1, 3], "w": [1, 3], "over": [1, 3], "all": [1, 3], "thu": [1, 3], "ik": 1, "onli": [1, 3], "one": [1, 3, 4], "w_0": 1, "offset": 1, "term": 1, "veri": [1, 3], "often": 1, "time": [1, 3], "mani": 1, "specimen": 1, "laboratori": [1, 3], "mean": [1, 3], "offer": 1, "wai": 1, "varieti": 1, "substanc": 1, "input": [1, 3], "unit": 1, "nm": 1, "associ": 1, "must": 1, "monoton": 1, "increas": 1, "central": [1, 3], "space": [1, 3], "numer": 1, "integr": 1, "w_prime": 1, "should": [1, 3], "small": 1, "enough": 1, "converg": 1, "respect": [1, 3], "normal": 1, "simpli": [1, 3, 4], "output": [1, 3], "contain": 1, "w_norm": 1, "w_norm_0": 1, "f_k": 1, "interp1d": 1, "flip": 1, "kind": 1, "cubic": 1, "n_prime": 1, "zero": 1, "cc": 1, "0": [1, 3], "w_norm_ev": 1, "go": [1, 3], "have": [1, 3], "3": [1, 3], "rang": 1, "w_norm_p1": 1, "min": 1, "w_norm_p2": 1, "max": [1, 3], "sum1": 1, "sum2": 1, "sum3": 1, "first": [1, 3], "n_1": 1, "int": 1, "ceil": 1, "w_norm_prime_1": 1, "linspac": [1, 3], "k_hb_prime_1": 1, "integrand_1": 1, "trapz": 1, "x": [1, 3], "second": [1, 3], "n_2": 1, "w_norm_prime_2": 1, "k_hb_prime_2": 1, "integrand_2": 1, "third": [1, 3], "n_3": 1, "w_norm_prime_3": 1, "k_hb_prime_3": 1, "integrand_3": 1, "pi": [1, 3], "return": 1, "ep": 1, "p_set": 1, "size": [1, 3], "alpha": [1, 3], "beta": [1, 3], "sigma": 1, "1j": [1, 3], "eps_calc": 1, "m": [1, 3], "ab": [1, 3], "code": 1, "loadmat": 1, "hb_extinction_data": 1, "mat": 1, "squeez": 1, "hbo2": 1, "hb": 1, "150": 1, "liter": 1, "solut": 1, "303": 1, "64500": 1, "k_hb": 1, "1e": [1, 3], "7": [1, 3], "4": [1, 3], "k_hbo2": 1, "fig": [1, 3], "figur": [1, 3], "set_size_inch": [1, 3], "10": [1, 3], "plot": [1, 3], "label": [1, 3], "xlabel": [1, 3], "ylabel": [1, 3], "legend": [1, 3], "now": [1, 3], "discuss": 1, "800": 1, "300": [1, 3], "solutiond": 1, "current": 1, "scale": [1, 3], "factor": 1, "set": 1, "number": 1, "desir": 1, "point": [1, 3], "whole": 1, "resolut": [1, 3], "dw_int": 1, "5000": 1, "388": 1, "n_prime_o2": 1, "392": 1, "ylim": 1, "375": 1, "405": 1, "final": 1, "seri": [1, 3], "document": 1, "appendix": 1, "aim": 1, "insid": 1, "other": [1, 3, 4], "below": [1, 3], "cite": 1, "buckleynanoantennadesignenhanced2021": 1, "equival": 1, "frq": 1, "gam": 1, "sig": 1, "greater": 1, "than": 1, "450": 1, "entir": 1, "could": [1, 3], "easili": [1, 3], "extend": [1, 3], "y_min": 1, "400": 1, "y_rang": 1, "y_high": 1, "p0": 1, "arrai": 1, "92": 1, "003": 1, "08": 1, "8": [1, 3], "05": 1, "005": 1, "re": [1, 3], "arg": 1, "max_nfev": 1, "50000": 1, "xtol": 1, "eps_opt": 1, "14": [1, 3], "ax1": [1, 3], "add_subplot": [1, 3], "sqrt": [1, 3], "o": 1, "set_xlabel": [1, 3], "set_ylabel": [1, 3], "ax2": [1, 3], "imag": 1, "de": 1, "ad": [1, 3], "properti": [1, 3], "instanc": [1, 3], "mat_nam": 1, "paramet": 1, "print": [1, 3], "_rang": 1, "mp": 1, "freqrang": 1, "um_scal": 1, "str": 1, "_eps_inf": 1, "num": 1, "_frq": 1, "_gam": 1, "_sig": 1, "suscept": 1, "_susc": 1, "lorentziansuscept": 1, "_frq1": 1, "gamma": 1, "_gam1": 1, "_sig1": 1, "e_suscept": 1, "valid_freq_rang": 1, "hb_rang": 1, "hb_eps_inf": 1, "9204613356930944": 1, "hb_frq1": 1, "425739251299398": 1, "hb_gam1": 1, "18899320982566495": 1, "hb_sig1": 1, "0022859170192929194": 1, "hb_frq2": 1, "3068451311889477": 1, "hb_gam2": 1, "08748741374271483": 1, "hb_sig2": 1, "001867059599216536": 1, "hb_frq3": 1, "7948478593572692": 1, "hb_gam3": 1, "12395529040291549": 1, "hb_sig3": 1, "0004489168927026274": 1, "hb_susc": 1, "procedur": [1, 3], "abov": [1, 3], "eps_meas_o2": 1, "9": [1, 3], "5": [1, 3], "0005": 1, "res_o2": 1, "eps_opt_o2": 1, "squared_error": 1, "hbo2_rang": 1, "hbo2_eps_inf": 1, "9309747551301373": 1, "hbo2_frq1": 1, "665440135712604": 1, "hbo2_gam1": 1, "05531174295631055": 1, "hbo2_sig1": 1, "0012472226347621234": 1, "hbo2_frq2": 1, "406957892056221": 1, "hbo2_gam2": 1, "1467597503016222": 1, "hbo2_sig2": 1, "003194232393855132": 1, "hbo2_frq3": 1, "8486889261138144": 1, "hbo2_gam3": 1, "10013538512305616": 1, "hbo2_sig3": 1, "00032574995735006306": 1, "hbo2_frq4": 1, "7361480652434025": 1, "hbo2_gam4": 1, "04233333456840626": 1, "hbo2_sig4": 1, "00015979022951372748": 1, "hbo2_susc": 1, "lei": 1, "bi": 1, "ping": 1, "yang": [1, 3], "light": [1, 3], "scatter": 1, "biconcav": 1, "deform": 1, "invari": 1, "imbed": 1, "t": [1, 4], "matrix": 1, "journal": 1, "biomed": 1, "18": [1, 3], "055001": 1, "mai": [1, 3, 4], "2013": 1, "doi": [1, 3], "1117": 1, "jbo": 1, "dirk": 1, "j": [1, 3], "faber": 1, "mauric": 1, "c": [1, 3], "aalder": 1, "egbert": 1, "mik": 1, "brett": 1, "A": 1, "hooper": 1, "martin": 1, "van": 1, "gemert": 1, "ton": 1, "leeuwen": 1, "satur": 1, "depend": 1, "physic": [1, 3], "review": 1, "letter": [1, 3], "93": 1, "028102": 1, "juli": [1, 3], "2004": 1, "section": 1, "highlight": 1, "color": 1, "blue": 1, "about": [1, 2], "what": [1, 3], "done": 1, "green": 1, "quantit": [1, 3], "specif": 1, "interest": [1, 3], "1103": 1, "physrevlett": 1, "drew": 1, "bucklei": 1, "yujia": [1, 3], "yugu": 1, "keathlei": [1, 3], "karl": [1, 3], "berggren": [1, 3], "phillip": [1, 3], "d": [1, 3], "nanoantenna": [1, 4], "design": 1, "enhanc": [1, 3], "carrier": [1, 3], "envelop": [1, 3], "phase": [1, 3], "sensit": [1, 3], "josa": 1, "b": [1, 3], "38": 1, "c11": 1, "c21": 1, "septemb": 1, "2021": [1, 3], "1364": [1, 3], "josab": 1, "424549": 1, "file": 1, "differ": [1, 3], "format": [1, 3], "csv": 1, "travel": [2, 3], "planar": 2, "interfac": 2, "while": [2, 3], "eventu": 2, "analysi": [2, 3, 4], "help": 2, "think": [2, 3], "explor": 3, "similarli": 3, "wire": 3, "captur": 3, "salient": 3, "dynam": 3, "effect": 3, "fact": 3, "each": 3, "s": 3, "damp": 3, "harmon": 3, "oscil": 3, "an": 3, "rlc": 3, "intuit": 3, "extern": 3, "drive": 3, "system": 3, "start": 3, "build": 3, "up": 3, "attach": 3, "via": 3, "studi": 3, "investig": 3, "impact": 3, "result": 3, "local": 3, "imagin": 3, "charg": 3, "structur": 3, "driven": 3, "excit": 3, "maximum": 3, "separ": 3, "capacit": 3, "between": 3, "top": 3, "bottom": 3, "furthermor": 3, "move": 3, "loss": 3, "induct": 3, "right": 3, "waveform": 3, "map": 3, "voltag": 3, "surround": 3, "interpret": 3, "correspond": 3, "capacitor": 3, "inde": 3, "written": 3, "v_c": 3, "frac": 3, "tild": 3, "v": 3, "l": 3, "r": 3, "mathrm": 3, "relat": 3, "respons": 3, "tip": 3, "which": 3, "express": 3, "_": 3, "propto": 3, "inc": 3, "omega_": 3, "tau": 3, "interestingli": 3, "might": [3, 4], "new": 3, "element": [3, 4], "would": 3, "make": 3, "situat": 3, "significantli": 3, "complic": 3, "consid": 3, "addit": 3, "shown": 3, "roughli": 3, "region": 3, "bodi": 3, "c_": 3, "gap": 3, "realiz": 3, "alreadi": 3, "simplifi": 3, "ignor": 3, "resist": 3, "itself": 3, "usual": 3, "concern": 3, "ourselv": 3, "applic": 3, "solv": 3, "v_": 3, "bit": 3, "work": [3, 4], "ac_": 3, "c_g": 3, "c_w": 3, "c_b": 3, "come": 3, "network": 3, "within": 3, "constant": 3, "multipl": 3, "modifi": 3, "strength": 3, "rewritten": 3, "our": 3, "problem": 3, "justifi": 3, "later": 3, "substitut": 3, "switch": 3, "despit": 3, "collector": 3, "seem": 3, "incred": 3, "power": 3, "sever": 3, "past": 3, "capict": 3, "tune": 3, "predict": 3, "tempor": 3, "profil": 3, "plasmon": [3, 4], "vari": 3, "bow": 3, "tie": 3, "well": 3, "transofrm": 3, "incid": 3, "paramt": 3, "fit": 3, "phenomenolog": 3, "comparison": 3, "data": 3, "wherev": 3, "possibl": 3, "choic": 3, "principl": 3, "individu": 3, "add": 3, "next": 3, "layer": 3, "coplex": 3, "These": [3, 4], "behav": 3, "wave": 3, "get": 3, "longer": 3, "cari": 3, "macroscop": 3, "distanc": 3, "induc": 3, "propag": 3, "delai": 3, "coudl": 3, "observ": 3, "across": 3, "pair": 3, "similar": 3, "instead": 3, "lump": [3, 4], "coupler": 3, "finit": 3, "length": 3, "sketch": 3, "devic": 3, "its": 3, "forward": 3, "backward": 3, "creat": 3, "reflect": 3, "togeth": 3, "outlin": 3, "select": 3, "v_1": 3, "v_2": 3, "proport": 3, "sert": 3, "close": 3, "match": 3, "capabl": 3, "cours": 3, "durat": 3, "For": 3, "natur": 3, "nanomet": 3, "femtosecond": 3, "relev": 3, "denot": 3, "subscript": 3, "tabl": 3, "f": 3, "12": 3, "h": 3, "epsilon_0": 3, "mu_0": 3, "pulse_funct": 3, "speed": 3, "eps0": 3, "85418782e": 3, "1e3": 3, "freespac": 3, "mu0": 3, "25663706e": 3, "6": 3, "1e9": 3, "permeabl": 3, "axi": 3, "1000": 3, "100000": 3, "fwhm": 3, "yc": 3, "1170": 3, "phasedelai": 3, "fc": 3, "wc": 3, "side": 3, "v1": 3, "cos2puls": 3, "v2": 3, "xlim": 3, "40": 3, "fontsiz": 3, "tick_param": 3, "labels": 3, "13": 3, "extinct": 3, "combin": 3, "recent": 3, "demonstr": 3, "abil": 3, "sampl": 3, "techniqu": 3, "given": 3, "case": 3, "valu": 3, "fem": 3, "unfortun": 3, "knowledg": 3, "doe": 3, "uniqu": 3, "estim": 3, "definit": 3, "lambda_": 3, "1100": 3, "af": 3, "intial": 3, "choos": 3, "ident": 3, "leav": 3, "plai": 3, "tau1": 3, "yres1": 3, "c1": 3, "10e": 3, "1e12": 3, "tau2": 3, "yres2": 3, "c2": 3, "wres1": 3, "l1": 3, "r1": 3, "wres2": 3, "l2": 3, "r2": 3, "2f": 3, "h_n": 3, "ohm_n": 3, "34055": 3, "6811": 3, "04": 3, "diretli": 3, "again": 3, "happen": 3, "yourself": 3, "chang": 3, "rel": 3, "them": 3, "dt": 3, "fft": 3, "fftfreq": 3, "v1f": 3, "vin1f": 3, "vin1": 3, "ifft": 3, "v2f": 3, "vin2f": 3, "vin2": 3, "15": 3, "set_xlim": 3, "20": 3, "set_titl": 3, "left": 3, "approxim": [3, 4], "perfect": 3, "serv": 3, "did": 3, "allow": 3, "imped": [3, 4], "z_0": 3, "equat": 3, "geometri": 3, "were": 3, "taken": 3, "clearli": 3, "formal": 3, "quit": 3, "bear": 3, "strike": 3, "resembl": 3, "200": 3, "width": 3, "500": 3, "epsr": 3, "3000": 3, "high": 3, "p1": 3, "amount": 3, "p2": 3, "lp": 3, "log": 3, "cp": 3, "split": 3, "rt1": 3, "rb1": 3, "lt1": 3, "lb1": 3, "zt1": 3, "zb1": 3, "rt2": 3, "rb2": 3, "lt2": 3, "lb2": 3, "zt2": 3, "zb2": 3, "veloc": 3, "z0": 3, "rad": 3, "293071": 3, "86": 3, "after": 3, "characterist": 3, "place": 3, "requir": 3, "gamma_l": 3, "effict": 3, "sourc": 3, "z_": 3, "back": 3, "telegraph": 3, "linear": 3, "do": 3, "twice": 3, "onc": 3, "off": 3, "total": 3, "when": 3, "activ": 3, "simultan": 3, "snippet": 3, "fo": 3, "calcualt": 3, "seen": 3, "zl1": 3, "gamma1": 3, "zin1": 3, "exp": 3, "zp1": 3, "vplus1f": 3, "vo21f": 3, "vg1f": 3, "vg21f": 3, "vg1": 3, "vg21": 3, "65": 3, "titl": 3, "16": 3, "zl2": 3, "gamma2": 3, "zin2": 3, "zp2": 3, "vplus2f": 3, "vo12f": 3, "vg2f": 3, "vg12f": 3, "vg2": 3, "vg12": 3, "sum": 3, "vg1t": 3, "vg2t": 3, "orthogon": 3, "polar": 3, "ellipt": 3, "exampl": 3, "90": 3, "degre": 3, "circularli": 3, "either": 3, "compon": 3, "direct": 3, "assum": 3, "outgo": 3, "ax": 3, "project": 3, "3d": 3, "view_init": 3, "135": 3, "plot3d": 3, "50": 3, "100": 3, "set_zlabel": 3, "bernd": 3, "metzger": 3, "mario": 3, "hentschel": 3, "marku": 3, "lippitz": 3, "harald": 3, "giessen": 3, "spectroscopi": 3, "nonlinear": 3, "37": 3, "22": 3, "4741": 3, "4743": 3, "novemb": 3, "2012": 3, "ol": 3, "004741": 3, "tobia": 3, "utik": 3, "emiss": 3, "spectrum": 3, "nano": 3, "3778": 3, "3782": 3, "1021": 3, "nl301686x": 3, "william": 3, "putnam": 3, "richard": 3, "hobb": 3, "k": 3, "franz": 3, "\u00e4": 3, "rtner": 3, "control": 3, "photoemiss": 3, "nanoparticl": 3, "335": 3, "339": 3, "april": 3, "2017": 3, "1038": 3, "nphys3978": 3, "vasireddi": 3, "vanish": 3, "11": 3, "1128": 3, "1133": 3, "2019": 3, "s41567": 3, "019": 3, "0613": 3, "mina": 3, "bionta": 3, "felix": 3, "ritzkowski": 3, "marco": 3, "turchetti": 3, "dario": 3, "cattozzo": 3, "mor": 3, "On": 3, "chip": 3, "attosecond": 3, "photon": 3, "456": 3, "460": 3, "june": 3, "s41566": 3, "021": 3, "00792": 3, "collect": 4, "my": 4, "hope": 4, "some": 4, "execut": 4, "style": 4, "notebook": 4, "handwritten": 4, "dai": 4, "convert": 4, "short": 4, "cabl": 4, "line": 4, "vs": 4, "kramer": 4, "kronig": 4, "refract": [], "index": [], "hemoglobin": 4, "red": 1, "blood": 1, "cell": 1, "surfac": 4, "polariton": 4, "mode": 4, "electr": 4, "coupl": 4, "reson": 4}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"short": 0, "cabl": 0, "imped": 0, "t": [0, 3], "line": [0, 3], "vs": 0, "lump": 0, "element": 0, "approxim": 0, "descript": [0, 1], "document": 0, "kramer": 1, "kronig": 1, "analysi": 1, "refract": 1, "index": 1, "hemoglobin": 1, "red": [], "blood": [], "cell": [], "introduct": 1, "header": 1, "function": 1, "calcul": [1, 3], "k": 1, "valu": 1, "from": [1, 3], "extinct": 1, "data": 1, "us": 1, "relat": 1, "drude": 1, "lorentz": 1, "oscil": 1, "fit": 1, "deoxygen": 1, "case": 1, "meep": 1, "materi": 1, "definit": 1, "oxygen": 1, "refer": [1, 3], "surfac": 2, "plasmon": 2, "polariton": 2, "mode": 2, "dispers": 2, "deriv": 2, "electr": 3, "coupl": 3, "nanoantenna": 3, "reson": 3, "motiv": 3, "circuit": 3, "model": 3, "transmiss": 3, "puls": 3, "setup": 3, "A": 3, "note": [3, 4], "unit": 3, "code": 3, "convers": 3, "si": 3, "nm": 3, "fs": 3, "follow": 3, "set": 3, "antenna": 3, "paramet": 3, "connect": 3, "With": 3, "initi": 3, "guess": 3, "twin": 3, "lead": 3, "technic": 4, "self": 4, "tabl": 4, "content": 4}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinxcontrib.bibtex": 9, "sphinx": 56}})