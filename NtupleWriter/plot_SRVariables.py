import ROOT

inFile = ROOT.TFile("ntuple_test.root")
tree = inFile.Get("Events")

variables = {
	"jet_n": ["Number of jets;n_{jets};Events",10, -0.5, 9.5],
	"bjet_n": ["Number of b-tagged jets;n_{b-jets};Events",10, -0.5, 9.5],
	"lep_n": ["Number of b-tagged jets;n_{b-jets};Events",10, -0.5, 9.5],
	"HT": ["H_{T};H_{T} [GeV];Events",20,0,1200],
	"deltaRLepClosestBJet": ["#DeltaR between the lepton and the closest b-jet;#DeltaR(l,b_{1})",40,0,5],
	"deltaRLep2ndClosestBJet": ["#DeltaR between the lepton and the second closest b-jet;#DeltaR(l,b_{2})",40,0,5],
	"LJet_m_plus_RCJet_m_12": ["Sum of the masses of the two leading reclustered jets with and without the lepton;m_{J^{lep}}+m_{J^{had}};Events",20,0,700],
	"bb_m_for_minDeltaR": ["Invariant mass of the two b-jets closest to each other;m_{bb#DeltaR_{min}} [GeV];Events",20,0,300]

}

c = ROOT.TCanvas("c","c",800,600)
c.SetLogy(1)
for var in variables:
	h = ROOT.TH1F("h_%s"%var,variables[var][0],variables[var][1],variables[var][2],variables[var][3])
	tree.Draw(var+">>h_%s"%var,"HT>300","e")
	h.GetYaxis().SetLabelSize(0.04)
	h.GetXaxis().SetLabelSize(0.04)
	h.GetYaxis().SetTitleSize(0.04)
	h.GetXaxis().SetTitleSize(0.04)
	h.SetBinContent(h.GetNbinsX(),h.GetBinContent(h.GetNbinsX())+h.GetBinContent(h.GetNbinsX()+1))
	h.GetYaxis().SetRangeUser(1,1e6)
	h.Draw()
	c.SaveAs("plots/SR/%s.pdf"%var)