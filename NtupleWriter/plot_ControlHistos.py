import ROOT

inFile = ROOT.TFile("ntuple.root")

histos =  ["n_jets","n_bjets", "n_electrons", "n_muons", "n_leptons","n_electrons_tight", "n_electrons_loose","n_muons_tight","n_muons_loose","n_leptons_tight","n_leptons_loose","cutflow"]

c = ROOT.TCanvas("c","c",800,600)
c.SetLogy(1)
for histo in histos:
	h = inFile.Get(histo)
	h.GetYaxis().SetLabelSize(0.04)
	h.GetXaxis().SetLabelSize(0.04)
	h.GetYaxis().SetTitleSize(0.04)
	h.GetXaxis().SetTitleSize(0.04)
	h.SetBinContent(h.GetNbinsX(),h.GetBinContent(h.GetNbinsX())+h.GetBinContent(h.GetNbinsX()+1))
	if histo == "cutflow":
		h.GetXaxis().SetBinLabel(1,"No sel.")
		h.GetXaxis().SetBinLabel(2,"n_{jets} #geq 4")
		h.GetXaxis().SetBinLabel(3,"n_{b-jets} #geq 2")
		h.GetXaxis().SetBinLabel(4,"n_{l}^{tight} == 1")
		h.GetXaxis().SetBinLabel(5,"n_{l}^{loose} == 0")
		print(h.GetBinContent(1))
	h.Draw()
	c.SaveAs("plots/control/%s.pdf"%histo)




