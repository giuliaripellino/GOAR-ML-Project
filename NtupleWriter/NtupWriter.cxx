/* 
Script to make flat ntuples from output from PhysObjectExtractor: https://github.com/cms-opendata-analyses/PhysObjectExtractorTool/tree/master/PhysObjectExtractor
Inspired by Olga's script: https://gitlab.cern.ch/darkmesonsearch/darkframework/-/blob/master/flatNTUPWriter1L/event_cycle.cxx#L284
Author: Giulia Ripellino giulia.ripellino@cern.ch
*/

#include "NtupWriter.h"
#include <ROOT/RVec.hxx>

void initialize() {

  return;
}

void lateInitialize() {
  char out_file_name[200];
  // sprintf(out_file_name, "ntuple_%s.root", output_tag.c_str());
  snprintf ( out_file_name, 200, "ntuple_%s.root", output_tag.c_str());
  out_file = new TFile(out_file_name, "NEW");
  out_tree = new TTree("Events","Events");

  declareOutputBranches();

  return;
}

void execute(int event) {
  // Get the event
  // std::cout << "Get event number " << event << std::endl;
  evt_tree->GetEntry(event);
 
  if (event == 0) lateInitialize();

  if (event%100000==0) {
    std::cout << "Processing event " << event << std::endl;
  } 

  // -----------------------------------------
  // LEPTON SELECTION 
  // based on pt, eta, isolation, quality 
  // -----------------------------------------

  std::vector<LepObj> Leptons;

  int mu_loose_n = 0;
  int mu_tight_n = 0;
  int mu_sel_n = 0;
  for (Long64_t i=0; i<mu_n;i++){
    if (!(mu_eta->at(i) > -2.5 && mu_eta->at(i) < 2.5)) continue;
    LepObj muon;
    muon.four_mom.SetPxPyPzE(mu_px->at(i),mu_py->at(i),mu_pz->at(i),mu_e->at(i));
    if (mu_isTight->at(i) && mu_pt->at(i) > 28 && mu_iso_rel->at(i)<0.15) {
      mu_sel_n++;
      muon.is_tight=true;
    }
    else if (mu_isLoose->at(i) && mu_pt->at(i) > 10) {
      muon.is_loose=true;
      mu_loose_n++;
    }
    mu_sel_n++;
    Leptons.push_back(muon);
  }

  int el_loose_n = 0;
  int el_tight_n = 0;
  int el_sel_n = 0;
  for (Long64_t i=0; i<el_n;i++){
    if (!(el_eta->at(i) > -2.5 && el_eta->at(i) < 2.5)) continue;
    LepObj electron;
    electron.four_mom.SetPxPyPzE(el_px->at(i),el_py->at(i),el_pz->at(i),el_e->at(i));
    if (el_isTight->at(i) && el_pt->at(i) > 28 && el_iso->at(i)/el_pt->at(i)<0.15) {
      electron.is_tight=true;
      el_tight_n++;
    }
    else if (el_isLoose->at(i) && el_pt->at(i) > 10) {
      electron.is_loose=true;
      el_loose_n++;
    }
    el_sel_n++;
    Leptons.push_back(electron);
  }

  // Require exactly one tight lepton
  h_mu_tight_n->Fill(mu_tight_n);
  h_el_tight_n->Fill(el_tight_n);
  h_lep_tight_n->Fill(mu_tight_n+el_tight_n);
  h_el_n->Fill(el_sel_n);
  h_mu_n->Fill(mu_sel_n);
  h_lep_n->Fill(el_sel_n+mu_sel_n);
  h_cutflow->Fill(1);

  if (!((mu_tight_n+el_tight_n)==1)) return;

  h_mu_loose_n->Fill(mu_loose_n); 
  h_el_loose_n->Fill(el_loose_n); 
  h_lep_loose_n->Fill(mu_loose_n+el_loose_n); 
  h_cutflow->Fill(2);

  // Veto additional loose leptons
  if (!((mu_loose_n+el_loose_n)==0)) return;
  h_cutflow->Fill(3);

  // Sort leptons by pt
  std::sort(std::begin(Leptons),std::end(Leptons),
            [](LepObj a,LepObj b){    
             return (a.four_mom.Pt() > b.four_mom.Pt());});

  // Get eta and phi of highest pt lepton and make sure that is the tight leptons, otherwise do not keep the event
  if (!(Leptons.at(0).is_tight)) return;
  h_cutflow->Fill(4);

  // -----------------------------------------
  // JET SELECTION
  // based on pt, eta, b-tag, overlap removal
  // -----------------------------------------

  std::vector<TLorentzVector> Jets;
  std::vector<TLorentzVector> BJets;
  for (Long64_t i=0; i<jet_n;i++){
    if (jet_pt->at(i)<20) continue;
    if (jet_eta->at(i)>2.5 || jet_eta->at(i)<-2.5) continue;
    TLorentzVector jet;
    jet.SetPtEtaPhiM(jet_pt->at(i),jet_eta->at(i),jet_phi->at(i),jet_m->at(i));
    float deltaR = Leptons.at(0).four_mom.DeltaR(jet);
    if (deltaR<0.4) continue; //This is a simple overlap removal
    if (jet_btag->at(i)>0.8){
      BJets.push_back(jet);
    }
    Jets.push_back(jet);
  }

  jet_sel_n = Jets.size();
  lep_sel_n = Leptons.size();
  bjet_sel_n = BJets.size();

  h_jet_n->Fill(jet_sel_n);
  // Require event to contain at least four jets
  if (jet_sel_n < 4) return;
  h_cutflow->Fill(5);

  h_bjet_n->Fill(bjet_sel_n);
  // Require event to contain two or more b-tagged jets
  if (bjet_sel_n < 2) return;
  h_cutflow->Fill(6);

  // std::cout << "Event with " << jet_sel_n << " jets, " << bjet_sel_n << " b-jets, " << mu_loose_n+el_loose_n << " loose leptons, " << mu_tight_n+el_tight_n << " tight leptons"<< std::endl;

  // -------------------------------------------
  // KINEMATIC VARIABLES based on standard jets
  // -------------------------------------------
  
  // HT
  float HT = 0.;
  for (Long64_t i=0; i<jet_sel_n;i++){
    HT += Jets.at(i).Pt();
  }
  ht = HT;

  // Recluster jets into R=1.2 jets
  PseudoJetVec LeptonPJs{};
  PseudoJetVec JetPJs{};
  PseudoJetVec Jets_12{};
  PseudoJetVec LJets_12{};

  // delta R between (highest-pT) Lepton and the second closest BJet to the lepton  
  deltaRLepClosestBJet = 999;
  deltaRLep2ndClosestBJet = 999;
  deltaR = 999;
  for (Long64_t i=0; i<bjet_sel_n;i++){
    deltaR = BJets.at(i).DeltaR(Leptons.at(0).four_mom);
    if ((deltaR>=deltaRLepClosestBJet) and (deltaR<deltaRLep2ndClosestBJet)){
      deltaRLep2ndClosestBJet=deltaR;
    }
    if (deltaR<deltaRLepClosestBJet){
      deltaRLep2ndClosestBJet = deltaRLepClosestBJet;
      deltaRLepClosestBJet = deltaR;
    }
  }

  // Minimum delta R between any b-jets
  deltaR = 999;
  minDeltaRBJets = 99999.9;
  int indexi;
  int indexj;
  for (Long64_t i=0; i<bjet_sel_n;i++){
    for (Long64_t j=i+1; j<bjet_sel_n;j++){
      deltaR = BJets.at(i).DeltaR(BJets.at(j));
      if (deltaR < minDeltaRBJets) { 
        minDeltaRBJets = deltaR; 
        indexi = i;  //use later to get invariant mass of closest bjet pair
        indexj = j;
        }
    }
  }

  // Minimum invariant mass of any 2 b-jets (bbm)
  min_m_bb = 9999.9;
  TLorentzVector bb_sum;
  for (Long64_t i=0; i<bjet_sel_n;i++){
    for (Long64_t j=i+1; j<bjet_sel_n;j++){      
       bb_sum = BJets.at(i)+BJets.at(j);
       if (bb_sum.M() < min_m_bb) min_m_bb = bb_sum.M();//smallest M bjet pair
       if ((i==indexi) and (j==indexj)) bb_m_for_minDeltaR = bb_sum.M(); //closest bjet pair
    }
  }


  for (Long64_t i=0; i<jet_sel_n;i++){
    JetPJs.push_back(fastjet::PseudoJet(Jets.at(i).Px(),Jets.at(i).Py(),Jets.at(i).Pz(),Jets.at(i).E()));
  }

  for (Long64_t i=0; i<(lep_sel_n);i++){
    LeptonPJs.push_back(fastjet::PseudoJet(Leptons.at(i).four_mom.Px(),Leptons.at(i).four_mom.Py(),Leptons.at(i).four_mom.Pz(),Leptons.at(i).four_mom.E()));
  }

  // --------------
  // RECLUSTERING 
  // --------------

  double jetrc_ptmin =  5; //from stop analysis 
  fastjet::JetDefinition jetDef{fastjet::antikt_algorithm,12/10.};
  auto JetPJs_tmp = JetPJs ; // copy ctor
  for (auto l: LeptonPJs) {
    JetPJs_tmp.push_back(l);
    JetPJs_tmp.back().set_user_index(-999);
  }
  auto cs = fastjet::ClusterSequence(JetPJs_tmp, jetDef);
  JetPJs_tmp = fastjet::sorted_by_pt(cs.inclusive_jets(jetrc_ptmin));
  for (auto jet: JetPJs_tmp) {
      int containsLepton = 0;
      for (auto c: jet.constituents()) {
          if (c.user_index() == -999) {containsLepton = -999;}
      }
      if (containsLepton == -999){LJets_12.push_back(jet);}
      else {Jets_12.push_back(jet);}
  }

  // Dave variable
  if (Jets_12.size()>0) { 
    LJet_m_plus_RCJet_m_12 = LJets_12.at(0).m() + Jets_12.at(0).m();
  }
  else {LJet_m_plus_RCJet_m_12=0;}


  // Fill the tree
  out_tree->Fill();

  return;
}

void finalize() {
  h_jet_n->Write();
  h_bjet_n->Write();
  h_el_n->Write();
  h_mu_n->Write();
  h_lep_n->Write();
  h_el_tight_n->Write();
  h_el_loose_n->Write();
  h_mu_tight_n->Write();
  h_mu_loose_n->Write();
  h_lep_tight_n->Write();
  h_lep_loose_n->Write();
  h_cutflow->Write();
  out_tree->Write();
  out_file->Close();

  return;
}