/* 
Macro to harmonise Delphes output with output from PhysObjectExtractor: https://github.com/cms-opendata-analyses/PhysObjectExtractorTool/tree/master/PhysObjectExtractor
Author: Giulia Ripellino giulia.ripellino@cern.ch
root -l fixupSignal.cxx'("input.root","output.root")'
*/

#include <iostream>
#include "TROOT.h"
#include "TTree.h"
#include "TFile.h"

#ifdef __CLING__
R__LOAD_LIBRARY(libDelphes)
#include "classes/DelphesClasses.h"
#include "external/ExRootAnalysis/ExRootTreeReader.h"
#include "external/ExRootAnalysis/ExRootResult.h"
#endif

class ExRootResult;
class ExRootTreeReader;

void fixupSignal(const char *inputFile="../../Samples/Signal/SU2L_35_500.CMS.Delphes.root", const char *outputFile="SU2L_35_500_flat.root")
{
    // Branches
    ULong64_t event_number;
    Int_t run_number;

    Int_t jet_n;
    std::vector<float>   jet_pt;
    std::vector<float>   jet_eta;
    std::vector<float>   jet_phi;
    std::vector<float>   jet_m;
    std::vector<double>   jet_btag;

    Int_t el_n;
    std::vector<float>   el_pt;
    std::vector<float>   el_eta;
    std::vector<float>   el_phi;
    std::vector<int>   el_isTight;
    std::vector<int>   el_isLoose;

    Int_t mu_n;
    std::vector<float>   mu_pt;
    std::vector<float>   mu_eta;
    std::vector<float>   mu_phi;
    std::vector<int>   mu_isTight;
    std::vector<int>   mu_isLoose;

    // Declare output tree
    TTree *out_tree = new TTree("Events","Events");
    out_tree->Branch("event", &event_number);
    out_tree->Branch("run", &run_number);

    out_tree->Branch("numberjet", &jet_n);
    out_tree->Branch("jet_btag", &jet_btag);
    out_tree->Branch("jet_pt", &jet_pt);
    out_tree->Branch("jet_eta", &jet_eta);
    out_tree->Branch("jet_phi", &jet_phi);
    out_tree->Branch("jet_mass", &jet_m);

    out_tree->Branch("numberelectron", &el_n);
    out_tree->Branch("electron_pt", &el_pt);
    out_tree->Branch("electron_eta", &el_eta);
    out_tree->Branch("electron_phi", &el_phi);
    out_tree->Branch("electron_isTight", &el_isTight);
    out_tree->Branch("electron_isLoose", &el_isLoose);

    out_tree->Branch("numbermuon", &mu_n);
    out_tree->Branch("muon_pt", &mu_pt);
    out_tree->Branch("muon_eta", &mu_eta);
    out_tree->Branch("muon_phi", &mu_phi);
    out_tree->Branch("muon_isTight", &mu_isTight);
    out_tree->Branch("muon_isLoose", &mu_isLoose);

    // Read Delphes file and fill ntuple
    gSystem->Load("libDelphes");

    TChain *chain = new TChain("Delphes");
    chain->Add(inputFile);

    ExRootTreeReader *treeReader = new ExRootTreeReader(chain);
    ExRootResult *result = new ExRootResult();

    // Get pointers to branches used in this analysis
    TClonesArray *branchEvent = treeReader->UseBranch("Event");
    TClonesArray *branchJet = treeReader->UseBranch("Jet");
    TClonesArray *branchElectron = treeReader->UseBranch("Electron");
    TClonesArray *branchMuon = treeReader->UseBranch("Muon");

    Long64_t allEntries = treeReader->GetEntries();
    
    Event *event;
    Jet *jet;
    Muon *muon;
    Electron *electron;

    // Loop over all events
    std::cout << "Starting loop over all events" << std::endl;
    for(Int_t entry = 0; entry < allEntries; ++entry) {
        // Load selected branches with data from specified event
        treeReader->ReadEntry(entry);

        event_number = entry;
        run_number = 9999;

        // Loop over all electrons in event
        Int_t i;
        el_n=branchElectron->GetEntries();
        for(i = 0; i < branchElectron->GetEntriesFast(); ++i) {
            electron = (Electron*) branchElectron->At(i);
            el_pt.push_back(electron->PT);
            el_eta.push_back(electron->Eta);
            el_phi.push_back(electron->Phi);
            el_isTight.push_back(1);
            el_isLoose.push_back(0);
        }
        mu_n=branchMuon->GetEntries();
        for(i = 0; i < branchMuon->GetEntriesFast(); ++i) {
            muon = (Muon*) branchMuon->At(i);
            mu_pt.push_back(muon->PT);
            mu_eta.push_back(muon->Eta);
            mu_phi.push_back(muon->Phi);
            mu_isTight.push_back(1);
            mu_isLoose.push_back(0);
        }
        jet_n=branchJet->GetEntries();
        for(i = 0; i < branchJet->GetEntriesFast(); ++i) {
            jet = (Jet*) branchJet->At(i);
            jet_pt.push_back((float)jet->PT);
            jet_eta.push_back((float)jet->Eta);
            jet_phi.push_back((float)jet->Phi);
            jet_m.push_back((float)jet->Mass);
            jet_btag.push_back((double)jet->BTag);
        }
        out_tree->Fill();

        mu_pt.clear();
        mu_eta.clear();
        mu_phi.clear();
        el_pt.clear();
        el_eta.clear();
        el_phi.clear();
        jet_pt.clear();
        jet_eta.clear();
        jet_phi.clear();
        jet_m.clear();
        jet_btag.clear();

    }
    TFile *out_file = new TFile(outputFile, "NEW");
    out_tree->Write();

    cout << "** Exiting..." << endl;

    delete treeReader;
    delete chain;
}