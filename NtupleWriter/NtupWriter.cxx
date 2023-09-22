// Script to make flat ntuples from output from PhysObjectExtractor: https://github.com/cms-opendata-analyses/PhysObjectExtractorTool/tree/master/PhysObjectExtractor
// Inspired by Olga's script: https://gitlab.cern.ch/darkmesonsearch/darkframework/-/blob/master/flatNTUPWriter1L/event_cycle.cxx#L284

#include "NtupWriter.h"
#include <ROOT/RVec.hxx>

void initialize() {

  return;
}


void lateInitialize() {

  out_file = new TFile("ntuple.root", "NEW");
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

  // Find b-tagged jets
  // Use a "medium" working point with combinedInclusiveSecondaryVertexV2BJetTags > 0.8
  std::vector<float>   bjet_pt;
  std::vector<float>   bjet_eta;
  std::vector<float>   bjet_phi;
  std::vector<float>   bjet_m;

  std::vector<float>   jet_sel_pt;
  std::vector<float>   jet_sel_eta;
  std::vector<float>   jet_sel_phi;
  std::vector<float>   jet_sel_m;

  int jet_sel_n_tmp = 0;
  int bjet_n_tmp = 0;
  for (Long64_t i=0; i<jet_n;i++){
    if (jet_pt->at(i)<20) continue;
    if (jet_eta->at(i)>2.5 || jet_eta->at(i)<-2.5) continue;
    jet_sel_n_tmp++;
    jet_sel_pt.push_back(jet_pt->at(i));
    jet_sel_eta.push_back(jet_eta->at(i));
    jet_sel_phi.push_back(jet_phi->at(i));
    jet_sel_m.push_back(jet_m->at(i));
    if (jet_btag->at(i)>0.8){
      bjet_n_tmp++;
      bjet_pt.push_back(jet_pt->at(i));
      bjet_eta.push_back(jet_eta->at(i));
      bjet_phi.push_back(jet_phi->at(i));
      bjet_m.push_back(jet_m->at(i));
    }
  }

  bjet_n = bjet_n_tmp;
  jet_sel_n = jet_sel_n_tmp;

  h_jet_n->Fill(jet_sel_n);
  h_mu_n->Fill(mu_n);
  h_el_n->Fill(el_n);
  h_lep_n->Fill(mu_n+el_n);
  h_cutflow->Fill(1);

  // Require event to contain at least four jets
  if (jet_sel_n < 4) return;
  h_cutflow->Fill(2);

  h_bjet_n->Fill(bjet_n);
  // Require event to contain two or more b-tagged jets
  if (bjet_n < 2) return;
  h_cutflow->Fill(3);

  // Now, require the event to contain exactly one tight lepton
  int mu_loose_n = 0;
  int mu_tight_n = 0;
  for (Long64_t i=0; i<mu_n;i++){
    if (mu_isTight->at(i) && mu_pt->at(i) > 28 && mu_iso_rel->at(i)<0.045) mu_tight_n++;
    else if (mu_isLoose->at(i) && mu_pt->at(i) > 10) mu_loose_n++;
  }

  int el_loose_n = 0;
  int el_tight_n = 0;
  for (Long64_t i=0; i<el_n;i++){
    if (el_isTight->at(i) && el_pt->at(i) > 28 && el_iso->at(i)/el_pt->at(i)<0.015) el_tight_n++;
    else if (el_isLoose->at(i) && el_pt->at(i) > 10) el_loose_n++;
  }

  h_mu_tight_n->Fill(mu_tight_n);
  h_el_tight_n->Fill(el_tight_n);
  h_lep_tight_n->Fill(mu_tight_n+el_tight_n);


  //Exactly one tight lepton
  if (!((el_tight_n+mu_tight_n)==1)) return;
  h_cutflow->Fill(4);

  h_mu_loose_n->Fill(mu_loose_n); 
  h_el_loose_n->Fill(el_loose_n); 
  h_lep_loose_n->Fill(mu_loose_n+el_loose_n); 

  //No additional loose leptons
  if (!((el_loose_n+mu_loose_n)==0)) return;
  h_cutflow->Fill(5);

  // std::cout << "Event with " << jet_n << " jets, " << bjet_n << "b-jets, " << el_loose_n << "/" << mu_loose_n << " loose el/mu and " << el_tight_n << "/" << mu_tight_n << " tight el/mu" << std::endl;

  // compute HT
  float HT = 0.;
  for (Long64_t i=0; i<jet_sel_n;i++){
    HT += jet_sel_pt.at(i);
  }
  ht = HT;

  // Recluster jets into R=1.2 jets
  JetVec Leptons{};
  JetVec Jets{};
  JetVec Jets_12{};
  JetVec LJets_12{};

  TLorentzVector tmp_j{};
  for (Long64_t i=0; i<jet_sel_n;i++){
    tmp_j.SetPtEtaPhiM(jet_sel_pt.at(i),jet_sel_eta.at(i),jet_sel_phi.at(i),jet_sel_m.at(i));
    Jets.push_back(fastjet::PseudoJet(tmp_j.Px(),tmp_j.Py(),tmp_j.Pz(),tmp_j.E()));
  }

  for (Long64_t i=0; i<el_n;i++){
    Leptons.push_back(fastjet::PseudoJet(el_px->at(i),el_py->at(i),el_pz->at(i),el_e->at(i)));
  }

  for (Long64_t i=0; i<mu_n;i++){
    Leptons.push_back(fastjet::PseudoJet(mu_px->at(i),mu_py->at(i),mu_pz->at(i),mu_e->at(i)));
  }

  //Sort by pt of lepton
  std::sort(std::begin(Leptons),std::end(Leptons),
            [](fastjet::PseudoJet a,fastjet::PseudoJet b){    
             return (a.pt() > b.pt());});

  double jetrc_ptmin =  5; //from stop analysis 
  fastjet::JetDefinition jetDef{fastjet::antikt_algorithm,12/10.};
  auto Jets_tmp = Jets ; // copy ctor
  for (auto l: Leptons) {
    Jets_tmp.push_back(l);
    Jets_tmp.back().set_user_index(-999);
  }
  auto cs = fastjet::ClusterSequence(Jets_tmp, jetDef);
  Jets_tmp = fastjet::sorted_by_pt(cs.inclusive_jets(jetrc_ptmin));
  for (auto jet: Jets_tmp) {
      int containsLepton = 0;
      for (auto c: jet.constituents()) {
          if (c.user_index() == -999) {containsLepton = -999;}
      }
      if (containsLepton == -999){LJets_12.push_back(jet);}
      else {Jets_12.push_back(jet);}
  }

  //delta R between (highest-pT) Lepton and the second closest BJet to the lepton  
  float lepton_eta = Leptons.at(0).eta();
  float lepton_phi = Leptons.at(0).phi(); 
  deltaRLepClosestBJet = 999;
  deltaRLep2ndClosestBJet = 999;
  deltaR = 999;
  for (Long64_t i=0; i<bjet_n;i++){
    deltaR = ROOT::VecOps::DeltaR(lepton_eta,bjet_eta.at(i),lepton_phi, bjet_phi.at(i));
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
  for (Long64_t i=0; i<bjet_n;i++){
    for (Long64_t j=i+1; j<bjet_n;j++){
      deltaR = ROOT::VecOps::DeltaR(bjet_eta.at(i), bjet_eta.at(j), bjet_phi.at(i), bjet_eta.at(j));
      if (deltaR < minDeltaRBJets) { 
        minDeltaRBJets = deltaR; 
        indexi = i;  //use later to get invariant mass of closest bjet pair
        indexj = j;
        }
    }
  }

  // Minimum invariant mass of any 2 b-jets (bbm)
  min_m_bb = 9999.9;
  ROOT::Math::PtEtaPhiMVector b1;
  ROOT::Math::PtEtaPhiMVector b2;
  ROOT::Math::PtEtaPhiMVector bb_sum;
  for (Long64_t i=0; i<bjet_n;i++){
    for (Long64_t j=i+1; j<bjet_n;j++){

       b1.SetPt(bjet_pt.at(j));
       b1.SetEta(bjet_eta.at(j));
       b1.SetPhi(bjet_phi.at(j));
       b1.SetM(bjet_m.at(j));

       b2.SetPt(bjet_pt.at(i));
       b2.SetEta(bjet_eta.at(i));
       b2.SetPhi(bjet_phi.at(i));
       b2.SetM(bjet_m.at(i));

      
       bb_sum = b1+b2;
       if (bb_sum.M() < min_m_bb) min_m_bb = bb_sum.M();//smallest M bjet pair
       if ((i==indexi) and (j==indexj)) bb_m_for_minDeltaR = bb_sum.M(); //closest bjet pair
    }
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

