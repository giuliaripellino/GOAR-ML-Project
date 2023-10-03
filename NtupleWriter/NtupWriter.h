#include <iostream>
#include "TROOT.h"
#include "TTree.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TSystem.h"
#include "TLorentzVector.h"
#include <Math/Vector4D.h>
#include <math.h>
#include <stdio.h>

#include <fastjet/tools/Filter.hh>
#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"

typedef std::vector<fastjet::PseudoJet> PseudoJetVec;


// Class for easy handling of leptons
class LepObj {
  public:
    TLorentzVector four_mom;
    bool is_tight = false;
    bool is_loose = false;
};


TTree *evt_tree = nullptr;
TTree *jet_tree = nullptr;
TTree *el_tree = nullptr;
TTree *mu_tree = nullptr;

// Declare functions
void initialize();
void lateInitialize();
void execute(int event);
void finalize();
void setBranches();

TFile *out_file;
TTree *out_tree;
std::string output_tag;

// Main function for steering the whole show
int main(int argc, char **argv) {

  int maxEvents = atoi(argv[3]);

  output_tag = argv[2];

  TFile *file = TFile::Open(argv[1]);
  if (!file->IsOpen()) {
  printf("DISASTER! Input file %s could not be opened. Exiting...\n", argv[0]);
	exit (EXIT_FAILURE);
  }

  std::cout << "After opening file" << std::endl;
  // Get nominal tree
  file->GetObject("myevents/Events", evt_tree);
  file->GetObject("myjets/Events", jet_tree);
  file->GetObject("myelectrons/Events", el_tree);
  file->GetObject("mymuons/Events", mu_tree);
  if (!( (evt_tree->GetEntries() == jet_tree->GetEntries()) && (jet_tree->GetEntries() == el_tree->GetEntries()) && (el_tree->GetEntries() == mu_tree->GetEntries()) ) ) {
    printf("DISASTER! Different number of events in trees. Exiting...\n");
    exit (EXIT_FAILURE);
  }
  if (evt_tree->GetEntries() == 0) return 1; 

  std::cout << "Now friending the trees" << std::endl;

  // Friend the trees
  evt_tree->AddFriend(jet_tree);
  evt_tree->AddFriend(el_tree);
  evt_tree->AddFriend(mu_tree);

  std::cout << "Setting branches" << std::endl;
  setBranches();

  initialize();
  
  if (maxEvents < 0) maxEvents = evt_tree->GetEntries();

  std::cout << "Starting event loop over " << maxEvents << " events" << std::endl;
  // Looping over events
  for (int i = 0; i < maxEvents; ++i) {
        execute(i);
  }

  finalize();

  file->Close();

  std::cout << "Done. Bye Bye!" << std::endl;

  return 0;
}

// Histograms
TH1* h_jet_n = new TH1D("n_jets", "Number of jets (N-1);n_{jets};Events", 10, -0.5, 9.5);
TH1* h_el_n = new TH1D("n_electrons", "Number of electrons;n_{e};Events", 10, -0.5, 9.5);
TH1* h_mu_n = new TH1D("n_muons", "Number of muons;n_{#mu};Events", 10, -0.5, 9.5);
TH1* h_lep_n = new TH1D("n_leptons", "Number of leptons;n_{l};Events", 10, -0.5, 9.5);
TH1* h_bjet_n = new TH1D("n_bjets", "Number of b-tagged jets (N-1);n_{b-jets};Events", 10, -0.5, 9.5);
TH1* h_el_tight_n = new TH1D("n_electrons_tight", "Number of tight electrons (N-1);n_{e} (tight);Events", 10, -0.5, 9.5);
TH1* h_el_loose_n = new TH1D("n_electrons_loose", "Number of loose electrons (N-1);n_{e} (loose);Events", 10, -0.5, 9.5);
TH1* h_mu_tight_n = new TH1D("n_muons_tight", "Number of tight muons (N-1);n_{#mu} (tight);Events", 10, -0.5, 9.5);
TH1* h_mu_loose_n = new TH1D("n_muons_loose", "Number of loose muons (N-1);n_{#mu} (loose);Events", 10, -0.5, 9.5);
TH1* h_lep_tight_n = new TH1D("n_leptons_tight", "Number of tight leptons (N-1);n_{lep} (tight);Events", 10, -0.5, 9.5);
TH1* h_lep_loose_n = new TH1D("n_leptons_loose", "Number of loose leptons (N-1);n_{lep} (loose);Events", 10, -0.5, 9.5);
TH1* h_cutflow = new TH1D("cutflow", "cutflow;cut;Events", 6, 0.5, 6.5);

// Branches
ULong64_t event_number;
Int_t run_number;
UInt_t lb_number;

Int_t jet_n;
std::vector<float>   *jet_pt;
std::vector<float>   *jet_eta;
std::vector<float>   *jet_phi;
std::vector<float>   *jet_m;
std::vector<double>   *jet_btag;

Int_t el_n;
std::vector<float>   *el_pt;
std::vector<float>   *el_px;
std::vector<float>   *el_py;
std::vector<float>   *el_pz;
std::vector<float>   *el_eta;
std::vector<float>   *el_phi;
std::vector<float>   *el_e;
std::vector<int> *el_isTight;
std::vector<int> *el_isLoose;
std::vector<float>   *el_iso; 
Int_t mu_n;
std::vector<float>   *mu_pt;
std::vector<float>   *mu_px;
std::vector<float>   *mu_py;
std::vector<float>   *mu_pz;
std::vector<float>   *mu_eta;
std::vector<float>   *mu_phi;
std::vector<float>   *mu_e;
std::vector<int> *mu_isTight;
std::vector<int> *mu_isLoose;
std::vector<float> *mu_iso_rel;

void setBranches() {
  evt_tree->SetBranchAddress("event", &event_number);
  evt_tree->SetBranchAddress("run", &run_number);
  evt_tree->SetBranchAddress("luminosityBlock", &lb_number);

  evt_tree->SetBranchAddress("numberjet", &jet_n);
  evt_tree->SetBranchAddress("jet_btag", &jet_btag);
  evt_tree->SetBranchAddress("jet_pt", &jet_pt);
  evt_tree->SetBranchAddress("jet_eta", &jet_eta);
  evt_tree->SetBranchAddress("jet_phi", &jet_phi);
  evt_tree->SetBranchAddress("jet_mass", &jet_m);

  evt_tree->SetBranchAddress("numberelectron", &el_n);
  evt_tree->SetBranchAddress("electron_pt", &el_pt);
  evt_tree->SetBranchAddress("electron_px", &el_px);
  evt_tree->SetBranchAddress("electron_py", &el_py);
  evt_tree->SetBranchAddress("electron_pz", &el_pz);
  evt_tree->SetBranchAddress("electron_eta", &el_eta);
  evt_tree->SetBranchAddress("electron_phi", &el_phi);
  evt_tree->SetBranchAddress("electron_e", &el_e);
  evt_tree->SetBranchAddress("electron_isTight", &el_isTight);
  evt_tree->SetBranchAddress("electron_isLoose", &el_isLoose);
  evt_tree->SetBranchAddress("electron_iso", &el_iso);

  evt_tree->SetBranchAddress("numbermuon", &mu_n);
  evt_tree->SetBranchAddress("muon_pt", &mu_pt);
  evt_tree->SetBranchAddress("muon_px", &mu_px);
  evt_tree->SetBranchAddress("muon_py", &mu_py);
  evt_tree->SetBranchAddress("muon_pz", &mu_pz);
  evt_tree->SetBranchAddress("muon_eta", &mu_eta);
  evt_tree->SetBranchAddress("muon_phi", &mu_phi);
  evt_tree->SetBranchAddress("muon_e", &mu_e);
  evt_tree->SetBranchAddress("muon_isTight", &mu_isTight);
  evt_tree->SetBranchAddress("muon_isLoose", &mu_isLoose);
  evt_tree->SetBranchAddress("muon_pfreliso03all", &mu_iso_rel);
}

// New branches
float ht;
Int_t bjet_sel_n;
Int_t jet_sel_n;
Int_t lep_sel_n;
float deltaRBJet1Lep;
float deltaRLepClosestBJet;
float deltaRLepClosestJet;
float deltaRLep2ndClosestBJet;
float minDeltaRBJets;
float deltaR; //Temporary variable
float min_m_bb;
float bb_m_for_minDeltaR;
float LJet_m_plus_RCJet_m_12;

void declareOutputBranches() {
  out_tree->Branch("event", &event_number, "event/I");
  out_tree->Branch("run", &run_number, "run/I");
  out_tree->Branch("lb", &lb_number, "lb/I");
  out_tree->Branch("jet_n", &jet_sel_n, "jet_n/I");
  out_tree->Branch("bjet_n", &bjet_sel_n, "bjet_n/I");
  out_tree->Branch("lep_n", &lep_sel_n, "lep_n/I");
  out_tree->Branch("HT", &ht, "HT/F");
  out_tree->Branch("deltaRLepClosestBJet", &deltaRLepClosestBJet, "deltaRLepClosestBJet/F");
  out_tree->Branch("deltaRLep2ndClosestBJet", &deltaRLep2ndClosestBJet, "deltaRLep2ndClosestBJet/F");
  out_tree->Branch("LJet_m_plus_RCJet_m_12", &LJet_m_plus_RCJet_m_12, "LJet_m_plus_RCJet_m_12/F");
  out_tree->Branch("bb_m_for_minDeltaR", &bb_m_for_minDeltaR, "bb_m_for_minDeltaR/F");
}

