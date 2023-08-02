#include "io.h"

void ReadInput(const char *input) {
    ifstream inf(input, ios::in);
    if(!inf.is_open()) { CERR << "no input file" << endl; EXIT(1); }

    // default values
    
    // vasp
    vaspgam = 0;
    vaspGAM = 0;
    vaspncl = 0;

    // auxiliary
    is_sub_calc = false;
    is_make_spinor = false;
    is_paw_calc = true;
    is_bse_calc = false;
    totdiffspns = 1;
    
    // basic
    taskmod = "fssh";
    carrier = "electron";
    dftcode = "vasp";
    vaspver = 6;
    vaspbin = "std";
    pawpsae = "ae";
    sexpikr = 0;
    ispinor = 0;
    memrank = 1;
    probind = -1;
    runhome = "run";
    ndigits = 4;
    totstru = 1;
    strubeg = -1;
    struend = -1;

    // basis sets
    numspns = 1;
    numkpts = 1;
    numbnds = 1;
    totbnds = 1;
    
    // dynamics
    hopmech = "nac";
    dyntemp = 100.0;
    iontime = 1.0;
    neleint = 10;
    intalgo = "Magnus";
    nsample = 0;
    ntrajec = 128;
    namdtim = 1;
    nvinidc = 1;

    // exciton
    epsilon = -1.0;
    encutgw = -1.0;
    iswfull = 0;
    gapdiff = -1.0;
    dynchan.resize(4, 1.0);
    lrecomb = 0;
    // auxiliary
    bsedone = 0;
    
    string        inputstr = WholeFile2String(inf);
    inf.close();
    istringstream inputss(inputstr);
    string line;
    string btopstr, bbotstr, bmaxstr, bminstr; // for band top/bottom/max/min
    string cmaxstr, cminstr, vmaxstr, vminstr; // for conduction/valence bands
    string exclstr; // for exclude
    string spnsstr, kptsstr;                   // for spins
    string readhpm; // read in hopmech
    string nkgwstr, nkscstr; // for nkpts gw/sc
    string dchnstr; // dynamics channel, for dynchan
    vector<string> vecstrtmp;
    while(getline(inputss, line)) {
        if(line.empty()) continue;
        vecstrtmp = StringSplitByBlank(line);

        // basic
             if(vecstrtmp[0] == "taskmod") taskmod =      vecstrtmp[2] ;
        else if(vecstrtmp[0] == "carrier") carrier =      vecstrtmp[2] ;
        else if(vecstrtmp[0] == "dftcode") dftcode =      vecstrtmp[2] ;
        else if(vecstrtmp[0] == "vaspver") vaspver = stoi(vecstrtmp[2]);
        else if(vecstrtmp[0] == "vaspbin") vaspbin =      vecstrtmp[2] ;
        else if(vecstrtmp[0] == "pawpsae") pawpsae =      vecstrtmp[2] ;
        else if(vecstrtmp[0] == "sexpikr") sexpikr = stoi(vecstrtmp[2]);
        else if(vecstrtmp[0] == "ispinor") ispinor = stoi(vecstrtmp[2]);
        else if(vecstrtmp[0] == "memrank") memrank = stoi(vecstrtmp[2]);
        else if(vecstrtmp[0] == "probind") probind = stoi(vecstrtmp[2]);
        else if(vecstrtmp[0] == "runhome") runhome =      vecstrtmp[2] ;
        else if(vecstrtmp[0] == "ndigits") ndigits = stoi(vecstrtmp[2]);
        else if(vecstrtmp[0] == "totstru") totstru = stoi(vecstrtmp[2]);
        else if(vecstrtmp[0] == "strubeg") strubeg = stoi(vecstrtmp[2]);
        else if(vecstrtmp[0] == "struend") struend = stoi(vecstrtmp[2]);

        // basis sets space
        else if(vecstrtmp[0] == "numspns") numspns = stoi(vecstrtmp[2]);
        else if(vecstrtmp[0] == "allspns") spnsstr =      line         ;
        else if(vecstrtmp[0] == "numkpts") numkpts = stoi(vecstrtmp[2]);
        else if(vecstrtmp[0] == "kpoints") kptsstr =      line         ;
        
        else if(vecstrtmp[0] == "bandtop") btopstr =      line         ;
        else if(vecstrtmp[0] == "bandmax") bmaxstr =      line         ;
        else if(vecstrtmp[0] == "bandmin") bminstr =      line         ;
        else if(vecstrtmp[0] == "bandbot") bbotstr =      line         ;
        
        else if(vecstrtmp[0] == "condmax") cmaxstr =      line         ;
        else if(vecstrtmp[0] == "condmin") cminstr =      line         ;
        else if(vecstrtmp[0] == "valemax") vmaxstr =      line         ;
        else if(vecstrtmp[0] == "valemin") vminstr =      line         ;
        
        else if(vecstrtmp[0] == "exclude") exclstr =      line         ;
        
        // dynamics
        else if(vecstrtmp[0] == "hopmech") readhpm =      vecstrtmp[2] ;
        else if(vecstrtmp[0] == "dyntemp") dyntemp = stod(vecstrtmp[2]);
        else if(vecstrtmp[0] == "iontime") iontime = stod(vecstrtmp[2]);
        else if(vecstrtmp[0] == "neleint") neleint = stoi(vecstrtmp[2]);
        else if(vecstrtmp[0] == "intalgo") intalgo =      vecstrtmp[2] ;
        else if(vecstrtmp[0] == "nsample") nsample = stoi(vecstrtmp[2]);
        else if(vecstrtmp[0] == "ntrajec") ntrajec = stoi(vecstrtmp[2]);
        else if(vecstrtmp[0] == "namdtim") namdtim = stoi(vecstrtmp[2]);
        else if(vecstrtmp[0] == "nvinidc") nvinidc = stoi(vecstrtmp[2]);

        // exciton
        else if(vecstrtmp[0] == "epsilon") epsilon = stod(vecstrtmp[2]);
        else if(vecstrtmp[0] == "wpotdir") wpotdir =      vecstrtmp[2] ;
        else if(vecstrtmp[0] == "iswfull") iswfull = stoi(vecstrtmp[2]);
        else if(vecstrtmp[0] == "encutgw") encutgw = stod(vecstrtmp[2]);
        else if(vecstrtmp[0] == "nkptsgw") nkgwstr =      line         ;
        else if(vecstrtmp[0] == "nkptssc") nkscstr =      line         ;
        else if(vecstrtmp[0] == "gapdiff") gapdiff = stod(vecstrtmp[2]);
        else if(vecstrtmp[0] == "dynchan") dchnstr =      line         ;
        else if(vecstrtmp[0] == "lrecomb") lrecomb = stoi(vecstrtmp[2]);

        // auxiliary
        else if(vecstrtmp[0] == "bsedone") bsedone = stoi(vecstrtmp[2]);
    }

    // task mode
    transform(taskmod.begin(), taskmod.end(), taskmod.begin(), ::tolower);
         if(taskmod == "bse") { carrier = "exciton"; is_bse_calc = true; }
    else if(taskmod == "spinor") { is_make_spinor = true; }
    else if(taskmod == "fssh" || taskmod == "dish" || taskmod == "dcsh") {}
    else { CERR << "NOT supported for this task: " << taskmod << endl; EXIT(1); }
    
    // dftcode
    transform(dftcode.begin(), dftcode.end(), dftcode.begin(), ::tolower);
    if(dftcode == "vasp") {
        transform(vaspbin.begin(), vaspbin.end(), vaspbin.begin(), ::tolower);
             if(vaspbin.find("std") != string::npos) { vaspbin = "std"; vaspgam = 0; vaspncl = 0; }
        else if(vaspbin.find("gam") != string::npos) { vaspbin = "gam"; vaspgam = 1; vaspncl = 0; numkpts = 1; }
        else if(vaspbin.find("ncl") != string::npos) { vaspbin = "ncl"; vaspgam = 0; vaspncl = 1; ispinor = 1; }
        else { CERR << "not surport for the vaspbin: " << vaspbin << endl; EXIT(1); }
    }
    else { CERR << "NOT surport for dftcode: " << dftcode << endl; EXIT(1); }

    // structures
    if(totstru != 0) {
        bool last_sub_calc = false;
        if(totstru < 0) last_sub_calc = true; // totstru will change later
        if(strubeg > 0 && struend == -1) { CERR << "\"strubeg\" setted, please set \"struend\"." << endl; EXIT(1); }
        if(struend > 0 && strubeg == -1) { CERR << "\"struend\" setted, please set \"strubeg\"." << endl; EXIT(1); }
        if(strubeg > struend) { CERR << "strubeg = " << strubeg << " shouldn't more than struend = " << struend << endl; EXIT(1); }
        if(strubeg == -1 && strubeg == -1) { strubeg = 1; struend = totstru; is_sub_calc = false; }
        else { totstru = struend - strubeg + 1; is_sub_calc = true; }
        if(is_sub_calc) { 
            if(last_sub_calc) laststru = struend;
            else laststru = struend + 1;
        }
        else laststru = totstru;
    }
    else {
        // combine previous calculations and modify totstru, strubeg, struend
    }
    
    // probind
    if(memrank >= 3 || probind > world_sz) probind = world_sz;
    else if(totstru < world_sz && probind == -1) probind = world_sz / totstru;
    else if(probind == -1) { // set default
        const int hardware_thread_num = thread::hardware_concurrency();
        const int thread_for_one_process = omp_get_num_procs();
        // binding all threads in one node
        if(world_sz * thread_for_one_process < hardware_thread_num) probind = world_sz;
        else probind = hardware_thread_num / thread_for_one_process;
    }
    
    // spins and spinor
    if(ispinor) { numspns = 1; Spins.push_back(0); }
    else if(numspns > 0 && numspns < 3) {
        if(spnsstr.empty()) for(int i = 0; i < numspns; i++) Spins.push_back(i);
        else {
            vecstrtmp = StringSplitByBlank(spnsstr);
            for(int i = 0; i < numspns; i++) Spins.push_back(stoi(vecstrtmp[2 + i]));
        }
    }
    else { numspns = 1; Spins.push_back(0); }
    if(numspns == 2 && Spins[0] > Spins[1]) { Spins[0] = 0; Spins[1] = 1; }
    if(dftcode == "vasp" && !vaspncl && ispinor) is_make_spinor = true;
    vaspGAM = (vaspgam && !is_make_spinor ? 1 : 0);
    if(dftcode == "vasp" && (pawpsae == "ae" || is_make_spinor)) is_paw_calc = true;

    // carrier and hopmech
    transform(carrier.begin(), carrier.end(), carrier.begin(), ::tolower);
         if(carrier.find("electron") != string::npos) carrier = "electron";
    else if(carrier.find("hole")     != string::npos) carrier = "hole";
    else if(carrier.find("exciton")  != string::npos) carrier = "exciton";
    else { CERR << "NOT surport for the carrier: " << carrier << endl; EXIT(1); }
    if(carrier == "electron" || carrier == "hole") {
        if(readhpm.empty()) hopmech = "nac";
        if(hopmech != "nac" && hopmech != "nacsoc") 
        { CERR << "NOT supported for this hopmech = " << hopmech << endl; EXIT(1); }
        if(hopmech == "nacsoc") {
            if(numspns == 1) {
                numspns = 2;
                Spins.push_back(Spins.back());
            }
        }
    }
    else if(carrier == "exciton") {
        if(readhpm.empty()) hopmech = "nacbse";
        else {
            transform(readhpm.begin(), readhpm.end(), readhpm.begin(), ::tolower);
            if(readhpm == "nacbse" || readhpm == "nac") hopmech = readhpm;
            else hopmech = "nacbse";
        }
        if(hopmech == "nacbse") is_bse_calc = true;
    }
    else { CERR << "NOT supported for this type of carrier" << endl; EXIT(1); }
    
    // spins and totdiffspns
    if(numspns == 1) totdiffspns = 1;
    else {
        if(Spins[1] == Spins[0]) totdiffspns = 1;
        else totdiffspns = 2;
    }
    
    // paw, ps or ae
    if(!(pawpsae == "ae" || pawpsae == "ps")) { CERR << "ONLY \"ps\" or \"ae\" supported for pawpsae" << endl; EXIT(1); }

    // kpoints
    if(numkpts > 0) { 
        if(vaspgam) Kpoints.push_back(1);
        else if(!kptsstr.empty()) {
            vecstrtmp = StringSplitByBlank(kptsstr);
            for(int i = 0; i < numkpts; i++) Kpoints.push_back(stoi(vecstrtmp[2 + i]));
        }
    }
    
    // btop, bbot, bmax, bmin, cmax, cmin, vmax, vmin
    if(carrier == "electron" || carrier == "hole") {
        if(bmaxstr.empty()) { CERR << "MUST input \"bandmax\"" << endl; EXIT(1); }
        vecstrtmp = StringSplitByBlank(bmaxstr); for(int i = 0; i < numspns; i++) bandmax.push_back(stoi(vecstrtmp[2 + i]));
        if(bminstr.empty()) { CERR << "MUST input \"bandmin\"" << endl; EXIT(1); }
        vecstrtmp = StringSplitByBlank(bminstr); for(int i = 0; i < numspns; i++) bandmin.push_back(stoi(vecstrtmp[2 + i]));
        
        if(btopstr.empty()) copy(bandmax.begin(), bandmax.end(), back_inserter(bandtop));
        else { vecstrtmp = StringSplitByBlank(btopstr); for(int i = 0; i < numspns; i++) bandtop.push_back(stoi(vecstrtmp[2 + i])); }
        if(bbotstr.empty()) copy(bandmin.begin(), bandmin.end(), back_inserter(bandbot));
        else { vecstrtmp = StringSplitByBlank(bbotstr); for(int i = 0; i < numspns; i++) bandbot.push_back(stoi(vecstrtmp[2 + i])); }

        for(int i = 0; i < numspns; i++) if( !(bandtop[i] >= bandmax[i] && bandmax[i] > bandmin[i] && bandmin[i] >= bandbot[i] && bandbot[i] > 0) ) { CERR << "WRONG for bands boundary" << endl; EXIT(1); }
        
        if(numspns == 2 && bandtop[0] - bandbot[0] != bandtop[1] - bandbot[1]) { CERR << "# of bands for spin up/down are not consistent: up-" << bandtop[0] - bandbot[0] << " down-" << bandtop[1] - bandbot[1] << endl; EXIT(1); }
        if(numspns == 2 && bandmax[0] - bandmin[0] != bandmax[1] - bandmin[1]) { CERR << "# of bands for spin up/down are not consistent: up-" << bandmax[0] - bandmin[0] << " down-" << bandmax[1] - bandmin[1] << endl; EXIT(1); }

        numbnds = bandtop[0] - bandbot[0] + 1;
        totbnds = bandmax[0] - bandmin[0] + 1;
    }
    else if(carrier == "exciton") {
        if(cmaxstr.empty()) { CERR << "MUST input \"condmax\"" << endl; EXIT(1); }
        vecstrtmp = StringSplitByBlank(cmaxstr); for(int i = 0; i < numspns; i++) condmax.push_back(stoi(vecstrtmp[2 + i]));
        if(cminstr.empty()) { CERR << "MUST input \"condmin\"" << endl; EXIT(1); }
        vecstrtmp = StringSplitByBlank(cminstr); for(int i = 0; i < numspns; i++) condmin.push_back(stoi(vecstrtmp[2 + i]));
        if(vmaxstr.empty()) { CERR << "MUST input \"valemax\"" << endl; EXIT(1); }
        vecstrtmp = StringSplitByBlank(vmaxstr); for(int i = 0; i < numspns; i++) valemax.push_back(stoi(vecstrtmp[2 + i]));
        if(vminstr.empty()) { CERR << "MUST input \"valemin\"" << endl; EXIT(1); }
        vecstrtmp = StringSplitByBlank(vminstr); for(int i = 0; i < numspns; i++) valemin.push_back(stoi(vecstrtmp[2 + i]));
        
        for(int i = 0; i < numspns; i++) if( !(condmax[i] >= condmin[i] && condmin[i] > valemax[i] && valemax[i] >= valemin[i] && valemin[i] > 0) ) { CERR << "WRONG for conduction or/and valence bands boundary" << endl; EXIT(1); }

        if(numspns == 2 && condmax[0] - condmin[0] != condmax[1] - condmin[1]) { CERR << "# of conduction bands for spin up/down are not consistent: up-" << condmax[0] - condmin[0] << " down-" << condmax[1] - condmin[1] << endl; EXIT(1); }
        if(numspns == 2 && valemax[0] - valemin[0] != valemax[1] - valemin[1]) { CERR << "# of valence bands for spin up/down are not consistent: up-"    << valemax[0] - valemin[0] << " down-" << valemax[1] - valemin[1] << endl; EXIT(1); }
        
        for(int i = 0; i < numspns; i++) { 
            bandtop.push_back(condmax[i]);
            bandbot.push_back(valemin[i]);
        }
        numbnds = (condmax[0] - condmin[0] + 1) + (valemax[0] - valemin[0] + 1);
        totbnds = (condmax[0] - valemin[0] + 1);
    }

    // exclude bands
    if(!exclstr.empty()) { 
        vecstrtmp = StringSplitByBlank(exclstr); 
        int itmp, num = 0;
        num = stoi(vecstrtmp[2]);
        exclude.push_back(num);
        for(int i = 0; i < numspns; i++) {
            for(int ib = 0; ib < num; ib++) {
                itmp = stoi(vecstrtmp[3 + i * num + ib]);
                if( ((carrier == "electron" || carrier == "hole") && itmp > bandmin[i] && itmp < bandmax[i]) ||
                    ( carrier == "exciton" && itmp > valemin[i] && itmp < valemax[i]) ||
                    ( carrier == "exciton" && itmp > condmin[i] && itmp < condmax[i]) ) exclude.push_back(itmp);
            }
        }
        if((exclude.size() - 1) % numspns) {CERR << "ERROR: # of bands excluded are different for spin up and down." << endl; EXIT(1); }
        exclude[0] = (exclude.size() - 1) / numspns;
        if(exclude[0] != num) COUT << "WARNING: input and legal # of exclude are not consistent, see details in \"output\"" << endl; MPI_Barrier(world_comm);
        numbnds -= exclude[0];
        if(carrier == "exciton") totbnds -= exclude[0];
    }
    else exclude.push_back(0);

    // exciton
    if(carrier == "exciton" && is_bse_calc == true) {
        if(epsilon < 0.0) {
            if(wpotdir.empty()) { CERR << "MUST input \"wpotdir\"" << endl; EXIT(1); }
            if(nkgwstr.empty()) { CERR << "MUST input \"nkptsgw\"" << endl; EXIT(1); }
            vecstrtmp = StringSplitByBlank(nkgwstr); for(int i = 0; i < 3; i++) nkptsgw.push_back(stoi(vecstrtmp[2 + i]));
        }
        else {
            iswfull = 0;
        }
        if(nkscstr.empty()) { CERR << "MUST input \"nkptssc\"" << endl; EXIT(1); }
        vecstrtmp = StringSplitByBlank(nkscstr); for(int i = 0; i < 3; i++) nkptssc.push_back(stoi(vecstrtmp[2 + i]));
        
        if(!dchnstr.empty()) {
            vecstrtmp = StringSplitByBlank(dchnstr);
            for(int i = 0; i < min<long>(vecstrtmp.size() - 2, 4); i++) dynchan[i] = stod(vecstrtmp[2 + i]);
        }
    }

    // dynamics
    neleint += neleint % 2; // guarantee even integral intervals
    if(intalgo != "Magnus" && intalgo != "Euler") intalgo = "Magnus"; // set to default
    ntrajec = (ntrajec / bckntrajs + 1) * bckntrajs; // only valid for dish/dcsh

    return;
}

template <typename T>
void Output1(ofstream &otf, const char *keywords, T value, const char *comment = "") {
    otf << "   " << std::left << setw(12) << setfill(' ') << keywords << "=     " << setw(12) << setfill(' ') << value << comment << endl;
    return;
}
void WriteOutput(const int flag, const char *info, const int inNKPTS, const char *output) {
    ofstream otf;
    if(flag) otf.open(output, ios::out|ios::app);
    else     otf.open(output, ios::out);
    if(!otf.is_open()) { CERR << "output file " << output << " not open" << endl; EXIT(1); }
    
    switch(flag) {
        case -1:
        { time_t now = time(NULL); otf << endl << " Finish at " << ctime(&now); }
        otf << " Totally cost " << info << " s";
        
        break;

        case 0: // basic
        otf << " ================= Parallel Hefei-NAMD 0.0.4 output =================" << endl << endl;
        otf << " Running on total " << world_sz << " process(es), "
            << omp_get_num_procs() * world_sz << " core(s), divided into " << mpi_split_num << " group(s)." << endl;
        { time_t now = time(NULL); otf << " Begin at " << ctime(&now) << endl; }
        Output1(otf, "taskmod", taskmod, "task mode choose, now supported: fssh, dish, dcsh, bse, spinor");
        Output1(otf, "carrier", carrier, "now supported: electron, hole, exciton");
        Output1(otf, "dftcode", dftcode, "now supported: vasp"); 
        if(dftcode == "vasp") {
            Output1(otf, "vaspver", vaspver, "2-5.2.x 4-5.4.x 6-6.x");
            Output1(otf, "vaspbin", vaspbin, "gam-gamma ncl-non-collinear std-standard");
        }
        Output1(otf, "pawpsae", pawpsae, "ps-pseudo ae-all-electron");
        Output1(otf, "sexpikr", sexpikr, "0-calc Ekrij in momory, 1-store Ekrij in disk");
        Output1(otf, "ispinor", ispinor, "0-Bloch 1-spinor");
        Output1(otf, "memrank", memrank, "machine memory rank: 1-high, 2-medium, >=3-low");
        Output1(otf, "probind", probind, "# of processes(usually nodes) binded to deal with one crystal structure");
        Output1(otf, "runhome", runhome, "all scf directories are listed in runhome");
        Output1(otf, "totstru", totstru, "total # of sturctures calculated in runhome");
        Output1(otf, "strubeg", strubeg, "structure begin #");
        Output1(otf, "struend", struend, "sturcture end   #, totstru~[strubeg, struend]");
        otf << endl;
        break;

        case 1: // basis sets
        Output1(otf, "numspns", numspns, "# of spin channels");
       {string strtmp = ""; for(int i = 0; i < numspns; i++) strtmp += to_string(Spins[i]) + ' ';
        Output1(otf, "allspns", strtmp);}
        Output1(otf, "numkpts", numkpts, "# of K-points");
       {string strtmp = ""; 
        if(inNKPTS > 0) for(int i = 0; i < numkpts; i++) strtmp += to_string(Kpoints[i]) + ' '; else strtmp = "ALL";
        Output1(otf, "kpoints", strtmp);}
        break;

        case 101: // electron/hole
       {string strtmp = ""; for(int i = 0; i < numspns; i++) strtmp += to_string(bandtop[i]) + ' ';
        Output1(otf, "bandtop", strtmp, "Hamiltionian/NAC storage top boundary");}
       {string strtmp = ""; for(int i = 0; i < numspns; i++) strtmp += to_string(bandmax[i]) + ' ';
        Output1(otf, "bandmax", strtmp, "band maximum index for dynamics");}
       {string strtmp = ""; for(int i = 0; i < numspns; i++) strtmp += to_string(bandmin[i]) + ' ';
        Output1(otf, "bandmin", strtmp, "band minimum index for dynamics");}
       {string strtmp = ""; for(int i = 0; i < numspns; i++) strtmp += to_string(bandbot[i]) + ' ';
        Output1(otf, "bandtop", strtmp, "Hamiltionian/NAC storage bottom boundary");}
        otf << endl;
        break;

        case 102: // exciton
       {string strtmp = ""; for(int i = 0; i < numspns; i++) strtmp += to_string(condmax[i]) + ' ';
        Output1(otf, "condmax", strtmp, "conduction band maximum index");}
       {string strtmp = ""; for(int i = 0; i < numspns; i++) strtmp += to_string(condmin[i]) + ' ';
        Output1(otf, "condmin", strtmp, "conduction band minimum index");}
       {string strtmp = ""; for(int i = 0; i < numspns; i++) strtmp += to_string(valemax[i]) + ' ';
        Output1(otf, "valemax", strtmp, "valence    band maximum index");}
       {string strtmp = ""; for(int i = 0; i < numspns; i++) strtmp += to_string(valemin[i]) + ' ';
        Output1(otf, "valemin", strtmp, "valence    band minimum index");}
        otf << endl;
        break;

        case 110:  // exclude
        for(int i = 0; i < numspns; i++) {
            string strtmp = to_string(exclude[0]) + ": ";
            for(int ib = 0; ib < exclude[0]; ib++) strtmp += to_string(exclude[1 + i * exclude[0] + ib]) + ' ';
            Output1(otf, "exclude", strtmp, ("band excluded for spin channel " + to_string(i)).c_str());
        }
        otf << endl;
        break;
        
        case 2: // dynamics
        Output1(otf, "hopmech", hopmech, "the hopping mechanism, now supported: nac, nacsoc, nacbse(only for exciton)");
        Output1(otf, "dyntemp", dyntemp, "the temperature of dynamics");
        Output1(otf, "iontime", iontime, "ionic step time, POTIM in vasp");
        Output1(otf, "neleint", neleint, "number of electronic integral, interval time = iontime / neleint");
        Output1(otf, "intalgo", intalgo, "electronic integral algorithm, so far support Magnus or Euler");
        Output1(otf, "nsample", nsample, "number of samples for dynamics");
        Output1(otf, "ntrajec", ntrajec, "number of trajectories for one sample");
        Output1(otf, "namdtim", namdtim, "namd time, should at least 2");
        Output1(otf, "nvinidc", nvinidc, "non-vanishing initial dynamic coefficients");
        break;

        case 3: // exciton
        Output1(otf, "epsilon", epsilon, "if < 0, use GW, else screen Coulomb potential = 1/epsilon * 1/r");
        if(epsilon < 0) {
            Output1(otf, "wpotdir", wpotdir);
            Output1(otf, "iswfull", iswfull, "0: only diagonal; 1: full wpot; default: 0");
            Output1(otf, "encutgw", encutgw, "default = -1; encutgw > 0 if there is ENCUTGW tag in vasp INCAR file");
            string strtmp = ""; for(int i = 0; i < 3; i++) strtmp += to_string(nkptsgw[i]) + ' ';
            Output1(otf, "nkptsgw", strtmp, "# of kpoints along three axes for GW");
        }
       {string strtmp = ""; for(int i = 0; i < 3; i++) strtmp += to_string(nkptssc[i]) + ' ';
        Output1(otf, "nkptssc", strtmp, "# of kpoints along three axes for BSE/dynamics");}
        Output1(otf, "bsedone", bsedone, "> 0 if bse was calculated before (not in this calculaion)");
        break;

        case 4: // exciton
        Output1(otf, "gapdiff", gapdiff, "gap difference between GW and DFT in primitive cell");
       {string strtmp = ""; for(int i = 0; i < 4; i++) strtmp += to_string(dynchan[i]) + ' ';
        Output1(otf, "dynchan", strtmp, "screen Coulumb, exchange, eph, and radiation scale");}
        Output1(otf, "lrecomb", lrecomb, "recombination option: 0-not recombine, 1-nonradiative, 2-radiative, 3-both");
        break;

        default: break;
    }

    otf.close();
}

void CreatWorkDir(int rk, int sz, MPI_Comm &comm) { // creat work directories
    int ires;
    if(is_world_root) {
        for(ires = 1; ires > 0; ires++) {
            resdir = "res_" + carrier + '_' + to_string(ires);
            if(access(resdir.c_str(), F_OK)) { // return > 0: no current resdir
                COUT << "result directory: " << resdir << endl;
                mkdir(resdir.c_str(), S_IRWXU);
                break;
            }
        }
    }
    MPI_Bcast(&ires, 1, MPI_INT, 0, comm);
    resdir = "res_" + carrier + '_' + to_string(ires); // broadcast to all processes
    return;
}

void CheckBasisSets(waveclass &wvc) {
    int itmp = 1;
    if(is_make_spinor) itmp = 2;
    if(bandtop.front() > itmp * wvc.totnbnds ||
       bandtop.back()  > itmp * wvc.totnbnds ) { CERR << "Input # of band top exceeds " << itmp * wvc.totnbnds << " which reads in wavefunction file." << endl; EXIT(1); }

    if(hopmech == "nac" || hopmech == "nacsoc") { // electron, hole, exciton(only nac)
        if(numkpts == 0) {
            numkpts = wvc.totnkpts;
            Kpoints.clear();
            for(int i = 1; i < numkpts + 1; i++) Kpoints.push_back(i);
        }
        else if(Kpoints.empty()) {
            Kpoints.push_back(1);
            numkpts = 1;
        }
    }
    else if(carrier == "exciton") {
        numkpts = wvc.totnkpts;
        Kpoints.clear();
        for(int i = 1; i < numkpts + 1; i++) Kpoints.push_back(i);
    }

    if(is_world_root) {
        if(carrier == "electron" || carrier == "hole") {
            WriteOutput(1); 
            WriteOutput(101);
        }
        else if(carrier == "exciton") {
            WriteOutput(1, NULL, hopmech == "nac" ? 1 : 0); 
            WriteOutput(102);
        }
        WriteOutput(110);
        if(carrier == "exciton" && is_bse_calc) WriteOutput(3);
        WriteOutput(2);
    }

    return;
}

bool CreatNAMDdir(waveclass &wvc) {
    if(carrier == "electron" || carrier == "hole") namddir = "namd_" + to_string(wvc.nbnds);
    else if(carrier == "exciton") namddir = "namd_" + to_string(wvc.nkpts) + 'x' + to_string(wvc.dimC) + 'x' + to_string(wvc.dimV);

    if(is_sub_calc) namddir += '_' + to_string(strubeg) + '_' + to_string(struend);

    int not_exist_namd_dir = 0;
    if(is_world_root) {
        not_exist_namd_dir = access(namddir.c_str(), F_OK);
        if(not_exist_namd_dir) {
            mkdir(namddir.c_str(), S_IRWXU);
            mkdir((namddir + "/tmpEnergy").c_str(), S_IRWXU);
            mkdir((namddir + "/tmpPhase").c_str(), S_IRWXU);
            if(carrier == "electron" || carrier == "hole") { 
                mkdir((namddir + "/tmpNAC").c_str(), S_IRWXU);
            }
            else if(carrier == "exciton") {
                mkdir((namddir + "/tmpDiagonal").c_str(), S_IRWXU);
                mkdir((namddir + "/tmpCBNAC").c_str(), S_IRWXU);
                mkdir((namddir + "/tmpVBNAC").c_str(), S_IRWXU);
                mkdir((namddir + "/tmpC2VNAC").c_str(), S_IRWXU);
                if(is_bse_calc) {
                    mkdir((namddir + "/tmpDirect").c_str(), S_IRWXU);
                    mkdir((namddir + "/tmpExchange").c_str(), S_IRWXU);
                    mkdir((namddir + "/.tmpEkrij").c_str(), S_IRWXU);
                }
            }
        }
        cout << "Output Files Store in " << namddir << endl;
    }
    MPI_Bcast(&not_exist_namd_dir, 1, MPI_INT, world_root, world_comm);

    return not_exist_namd_dir;
}

void ReadInfoTmp(int &nspns, int &nkpts, int &dimC, int &dimV, const char *infofile) {
    ifstream inf(infofile, ios::in|ios::binary);
    if(!inf.is_open()) { cerr << infofile << " can't open to read in io.cpp::ReadInfoTmp" << endl; exit(1); }
    inf.read((char*)&nspns, sizeof(int));
    inf.read((char*)&nkpts, sizeof(int));
    inf.read((char*)&dimC, sizeof(int));
    inf.read((char*)&dimV, sizeof(int));
    inf.close();
    return;
}

void ReadInfoTmp1(vector<int> &readspns, vector<int> &readkpts, vector<int> &readbnds, const char* infofile) {
    readspns.clear(); readkpts.clear(); readbnds.clear();
    ifstream inf(infofile, ios::in|ios::binary);
    if(!inf.is_open()) { cerr << infofile << " can't open to read in io.cpp::ReadInfoTmp1" << endl; exit(1); }
    inf.seekg(sizeof(int) * 4, ios::beg);
    int rdnspns, rdnkpts, rdnbnds;
    int spn, kpt, bnd;
    
    inf.read((char*)&rdnspns, sizeof(int));
    for(int is = 0; is < rdnspns; is++) {
        inf.read((char*)&spn, sizeof(int));
        readspns.push_back(spn);
    }

    inf.read((char*)&rdnkpts, sizeof(int));
    for(int ik = 0; ik < rdnkpts; ik++) {
        inf.read((char*)&kpt, sizeof(int));
        readkpts.push_back(kpt);
    }

    inf.read((char*)&rdnbnds, sizeof(int));
    for(int is = 0; is < rdnspns; is++)
    for(int ib = 0; ib < rdnbnds; ib++) {
        inf.read((char*)&bnd, sizeof(int));
        readbnds.push_back(bnd);
    }

    inf.close();

    return;
}

void ReadInfoTmp2(vector<int> &readspns, vector<int> &readkpts,
                  vector<int> &readcbds, vector<int> &readvbds, const char* infofile) {
    readspns.clear(); readkpts.clear();
    readcbds.clear(); readvbds.clear();
    ifstream inf(infofile, ios::in|ios::binary);
    if(!inf.is_open()) { cerr << infofile << " can't open to read in io.cpp::ReadInfoTmp2" << endl; exit(1); }
    inf.seekg(sizeof(int) * 4, ios::beg);
    int rdnspns, rdnkpts, rdncbs, rdnvbs;
    int spn, kpt, cbd, vbd;
    
    inf.read((char*)&rdnspns, sizeof(int));
    for(int is = 0; is < rdnspns; is++) {
        inf.read((char*)&spn, sizeof(int));
        readspns.push_back(spn);
    }

    inf.read((char*)&rdnkpts, sizeof(int));
    for(int ik = 0; ik < rdnkpts; ik++) {
        inf.read((char*)&kpt, sizeof(int));
        readkpts.push_back(kpt);
    }

    inf.read((char*)&rdncbs, sizeof(int));
    inf.read((char*)&rdnvbs, sizeof(int));
    for(int is = 0; is < rdnspns; is++) {
        for(int ic = 0; ic < rdncbs; ic++) {
            inf.read((char*)&cbd, sizeof(int));
            readcbds.push_back(cbd);
        }
        for(int iv = 0; iv < rdnvbs; iv++) {
            inf.read((char*)&vbd, sizeof(int));
            readvbds.push_back(vbd);
        }
    }

    inf.close();

    return;
}

void CheckIniconFile(vector<int> *&allbands, const char *inicon) {
    ifstream inf(inicon, ios::in);
    if(!inf.is_open()) { cout << "\"" << inicon << "\" file doesn't find, please check."; exit(1); }
    string line;
    vector<string> vecstrtmp;
    int spn, kpt, cbd, vbd, bnd;
    int ispn;
    double wght;
    int rndifspns, nkpts, dimC, dimV;
    ReadInfoTmp(rndifspns, nkpts, dimC, dimV);

    cout << "Checking " << inicon << " file >>>>>> " << flush;
    for(int ii = 0; ii < nsample; ii++) {
        if(!getline(inf, line)) {
            cerr << inicon << " file not complete, should at least nsample = " << nsample << " lines";
            exit(1);
        }
        vecstrtmp = StringSplitByBlank(line);
        for(int inv = 0; inv < nvinidc; inv++) {
            if(carrier == "electron" || carrier == "hole") {
                spn = stoi(vecstrtmp[1 + 4 * inv]);
                assert(spn < numspns); // because allspns may have repeated elements, here set spn = ispn
                ispn = spn;
                
                kpt = stoi(vecstrtmp[1 + 4 * inv + 1]); 
                assert( find(Kpoints.begin(), Kpoints.end(), kpt) != Kpoints.end() );
                
                bnd = stoi(vecstrtmp[1 + 4 * inv + 2]); 
                assert( find(allbands[ispn].begin(), allbands[ispn].end(), bnd) != allbands[ispn].end() );
                
                if(nvinidc > 1) { wght = stod(vecstrtmp[1 + 4 * inv + 3]); assert(wght > 0.0); }
            }
            else if(carrier == "exciton") {
                spn = stoi(vecstrtmp[1 + 5 * inv]);
                assert(spn < numspns); // because allspns may have repeated elements, here set spn = ispn
                ispn = spn;
                
                kpt = stoi(vecstrtmp[1 + 5 * inv + 1]); 
                assert( find(Kpoints.begin(), Kpoints.end(), kpt) != Kpoints.end() );
                
                cbd = stoi(vecstrtmp[1 + 5 * inv + 2]); 
                assert( find(allbands[ispn].begin(), allbands[ispn].begin() + dimC, cbd) 
                                                  != allbands[ispn].begin() + dimC ); 

                vbd = stoi(vecstrtmp[1 + 5 * inv + 3]); 
                assert( find(allbands[ispn].begin() + dimC, allbands[ispn].end(), vbd) 
                                                         != allbands[ispn].end() ); 
            }
        } // inv
    } // ii for sample
    inf.close();
    cout << "right." << endl;
    return;
}

void WritePdvec(ofstream &otf, const int dim, const int dim_loc,
                const double *vec, double *fullvec,
                const bool is_new_line) {
/* 
    "vec" is a distributed column vector, 
    "fullvec" is an auxiliary vector fully malloced in root process
*/
    Blacs_MatrixDGather(dim, 1, vec, dim_loc, fullvec, dim);
    if(is_sub_root) {
        for(int ii = 0; ii < dim; ii++) otf << setiosflags(ios::fixed) << setprecision(18) << fullvec[ii] << ' ';
        if(is_new_line) otf << endl;
    }
    MPI_Barrier(group_comm);
    return;
}

void WritePzvec(ofstream &otf, const int dim, const int dim_loc, 
                const complex<double> *vec, complex<double> *fullvec,
                const bool is_new_line) {
/* 
    "vec" is a distributed column vector
    "fullvec" is an auxiliary vector fully malloced in root process
*/
    Blacs_MatrixZGather(dim, 1, vec, dim_loc, fullvec, dim);
    if(is_sub_root) {
        for(int ii = 0; ii < dim; ii++) otf << setiosflags(ios::fixed) << setprecision(18) << fullvec[ii] << ' ';
        if(is_new_line) otf << endl;
    }
    MPI_Barrier(group_comm);
    return;
}
