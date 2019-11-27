import os, math, optparse, ROOT
from array import array

ROOT.gStyle.SetOptStat(111111)
ROOT.gROOT.SetBatch(True)

def doLegend(histos,labels,styles,corner="TR",textSize=0.035,legWidth=0.18,legBorder=False,nColumns=1):
    nentries = len(histos)
    (x1,y1,x2,y2) = (.85-legWidth, .7 - textSize*max(nentries-3,0), .90, .91)
    if corner == "TR":
        (x1,y1,x2,y2) = (.85-legWidth, .7 - textSize*max(nentries-3,0), .90, .91)
    elif corner == "TC":
        (x1,y1,x2,y2) = (.5, .75 - textSize*max(nentries-3,0), .5+legWidth, .91)
    elif corner == "TL":
        (x1,y1,x2,y2) = (.2, .75 - textSize*max(nentries-3,0), .2+legWidth, .91)
    elif corner == "BR":
        (x1,y1,x2,y2) = (.85-legWidth, .33 + textSize*max(nentries-3,0), .90, .15)
    elif corner == "BC":
        (x1,y1,x2,y2) = (.5, .33 + textSize*max(nentries-3,0), .5+legWidth, .35)
    elif corner == "BL":
        (x1,y1,x2,y2) = (.2, .33 + textSize*max(nentries-3,0), .33+legWidth, .35)
    leg = ROOT.TLegend(x1,y1,x2,y2)
    leg.SetNColumns(nColumns)
    leg.SetFillColor(0)
    leg.SetFillColorAlpha(0,0.6)  # should make the legend semitransparent (second number is 0 for fully transparent, 1 for full opaque)
    #leg.SetFillStyle(0) # transparent legend, so it will not cover plots (markers of legend entries will cover it unless one changes the histogram FillStyle, but this has other effects on color, so better not touching the FillStyle)
    leg.SetShadowColor(0)
    if not legBorder:
        leg.SetLineColor(0)
        leg.SetBorderSize(0)  # remove border  (otherwise it is drawn with a white line, visible if it overlaps with plots
    leg.SetTextFont(42)
    leg.SetTextSize(textSize)
    for (plot,label,style) in zip(histos,labels,styles): leg.AddEntry(plot,label,style)
    leg.Draw()
    ## assign it to a global variable so it's not deleted
    global legend_
    legend_ = leg
    return leg


def plotDensity():
    tf = ROOT.TFile('reco_run815.root')
    tree = tf.Get('Events')

    histos = []
    colors = [ROOT.kRed,ROOT.kBlue,ROOT.kOrange]
    #histo = ROOT.TH1F('density','',70,0,2500)
    histo = ROOT.TH1F('density','',70,0,20)
    for it in xrange(1,4):
        h = histo.Clone('h_iter{it}'.format(it=it))
        h.Sumw2()
        #tree.Draw('track_integral/track_nhits>>h_iter{it}'.format(it=it),'track_iteration=={it}'.format(it=it))
        #tree.Draw('track_integral/track_length>>h_iter{it}'.format(it=it),'track_iteration=={it}'.format(it=it))
        tree.Draw('track_length>>h_iter{it}'.format(it=it),'track_iteration=={it}'.format(it=it))
        h.Scale(1./h.Integral())
        h.SetFillColor(colors[it-1])
        h.SetLineColor(colors[it-1])
        h.SetFillStyle(3005)
        histos.append(h)


    # legend
    (x1,y1,x2,y2) = (0.7, .70, .9, .87)
    leg = ROOT.TLegend(x1,y1,x2,y2)
    leg.SetFillColor(0)
    leg.SetFillColorAlpha(0,0.6)
    leg.SetShadowColor(0)
    leg.SetLineColor(0)
    leg.SetBorderSize(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.035)
    
    c = ROOT.TCanvas('c','',600,600)
    for ih,h in enumerate(histos):
        h.Draw('hist' if ih==0 else 'hist same')
        h.GetYaxis().SetRangeUser(0,0.25)
        h.GetXaxis().SetTitle('length (mm)')
        leg.AddEntry(h,'iteration {it}'.format(it=ih+1),'f')

    leg.Draw()

    c.SaveAs('density.pdf')


def plotNClusters(iteration=1):

    f = ROOT.TFile.Open('../runs/fng_runs.root')
    nclu_h = ROOT.TH1F('nclu_h','',20,0,80)
    
    for ie,event in enumerate(f.Events):
        #if event.run<46 or event.run>47: continue
        #if event.run<48 or event.run>50: continue
        if event.run<70 or event.run>71: continue
        nclu_it1 = 0
        for icl in range(event.nTrack):
            if event.track_nhits[icl]<100: continue
            if int(event.track_iteration[icl])>2: continue
            if math.hypot(event.track_xmean[icl]-1024,event.track_ymean[icl]-1024)>1000: continue
            #print "cluster x = ", event.track_xmean[icl]
            #print "cluster y = ", event.track_ymean[icl]
            nclu_it1 += 1
        nclu_h.Fill(nclu_it1)

    c = ROOT.TCanvas('c1','',600,400)
    nclu_h.SetLineColor(ROOT.kRed+2)
    nclu_h.SetMarkerColor(ROOT.kBlack)
    nclu_h.SetMarkerSize(1.5)
    nclu_h.SetMarkerStyle(ROOT.kOpenCircle)
    nclu_h.SetLineWidth(2)
    nclu_h.GetXaxis().SetTitle('# clusters / 100 ms')
    nclu_h.GetYaxis().SetTitle('Events')
    nclu_h.Draw('pe')
    nclu_h.Draw('hist same')
    c.SaveAs("nclusters_iter1_run70_71.pdf")


def withinFC(x,y,ax=500,ay=500,shape=2048):
    center = shape/2.
    x1 = x-center
    y1 = (y-center)*ax/ay
    return math.hypot(x1,y1)<ax
    

def fillSpectra(cluster='cl'):

    ret = {}
    tf_cosmics = ROOT.TFile('runs/reco_run01726_3D.root')
    tf_fe55 = ROOT.TFile('runs/reco_run01740_3D.root')
    tfiles = {'fe':tf_fe55,'cosm':tf_cosmics}
    
    ## signal region histograms
    ret[('fe','integral')] = ROOT.TH1F("fe_integral",'',100,-10,8000)
    ret[('fe','length')]   = ROOT.TH1F("fe_length",'',100,0,500)
    ret[('fe','width')]    = ROOT.TH1F("fe_width",'',35,0,70)
    ret[('fe','nhits')]    = ROOT.TH1F("fe_nhits",'',100,0,1500)
    ret[('fe','slimness')] = ROOT.TH1F("fe_slimness",'',100,0,1)

    # x-axis titles
    titles = {'integral': 'photons', 'length':'length (pixels)', 'width':'width (pixels)', 'nhits': 'active pixels', 'slimness': 'width/length'}
    
    ## control region histograms
    ret2 = {}
    for (region,var),h in ret.iteritems():
        ret[(region,var)].Sumw2()
        ret[(region,var)].SetDirectory(None)
        ret[(region,var)].GetXaxis().SetTitle(titles[var])
        ret[(region,var)].GetXaxis().SetTitleSize(0.1)
        ret2[('cosm',var)] = h.Clone('cosm_{name}'.format(name=var))
        ret2[('cosm',var)].SetDirectory(None)
    ret.update(ret2)

    ## now fill the histograms 
    for runtype in ['fe','cosm']:
        for ie,event in enumerate(tfiles[runtype].Events):
            for isc in range(getattr(event,"nSc" if cluster=='sc' else 'nCl')):
                if getattr(event,"{clutype}_iteration".format(clutype=cluster))[isc]!=2:
                    continue
                if not withinFC(getattr(event,"{clutype}_xmean".format(clutype=cluster))[isc],getattr(event,"{clutype}_ymean".format(clutype=cluster))[isc]):
                    continue
                for var in ['integral','length','width','nhits']:
                    ret[(runtype,var)].Fill(getattr(event,("{clutype}_{name}".format(clutype=cluster,name=var)))[isc])
                ret[(runtype,'slimness')].Fill(getattr(event,"{clutype}_width".format(clutype=cluster))[isc] / getattr(event,"{clutype}_length".format(clutype=cluster))[isc])

    return ret

def getCanvas():
    c = ROOT.TCanvas('c','',1200,1200)
    lMargin = 0.14
    rMargin = 0.10
    bMargin = 0.15
    tMargin = 0.10
    c.SetLeftMargin(lMargin)
    c.SetRightMargin(rMargin)
    c.SetTopMargin(tMargin)
    c.SetBottomMargin(bMargin)
    c.SetFrameBorderMode(0);
    c.SetBorderMode(0);
    c.SetBorderSize(0);
    return c

def drawOne(histo_sr,histo_cr,plotdir='./'):
    ROOT.gStyle.SetOptStat(0)
    
    c = ROOT.TCanvas('c','',1200,1200)
    lMargin = 0.12
    rMargin = 0.05
    bMargin = 0.30
    tMargin = 0.07
    padTop = ROOT.TPad('padTop','',0.,0.4,1,0.98)
    padTop.SetLeftMargin(lMargin)
    padTop.SetRightMargin(rMargin)
    padTop.SetTopMargin(tMargin)
    padTop.SetBottomMargin(0)
    padTop.SetFrameBorderMode(0);
    padTop.SetBorderMode(0);
    padTop.SetBorderSize(0);
    padTop.Draw()

    padBottom = ROOT.TPad('padBottom','',0.,0.02,1,0.4)
    padBottom.SetLeftMargin(lMargin)
    padBottom.SetRightMargin(rMargin)
    padBottom.SetTopMargin(0)
    padBottom.SetBottomMargin(bMargin)
    padBottom.SetFrameBorderMode(0);
    padBottom.SetBorderMode(0);
    padBottom.SetBorderSize(0);
    padBottom.Draw()

    padTop.cd()
    histo_cr.SetMaximum(1.2*max(histo_cr.GetMaximum(),histo_sr.GetMaximum()))
    histo_cr.GetYaxis().SetTitle('clusters (a.u.)')
    histo_cr.Draw("hist")
    histo_sr.Draw("pe same")

    histos = [histo_sr,histo_cr]
    labels = ['^{55}Fe source','No source']
    styles = ['p','f']
    
    legend = doLegend(histos,labels,styles,corner="TR")
    legend.Draw()
    
    padBottom.cd()
    ratio = histo_sr.Clone(histo_sr.GetName()+"_ratio")
    ratio.Divide(histo_cr)
    ratio.GetYaxis().SetTitleSize(0.05)
    ratio.GetYaxis().SetTitle("{num} / {den}".format(num=labels[0],den=labels[1]))
    ratio.Draw('pe1')

    line = ROOT.TLine()
    line.DrawLine(ratio.GetXaxis().GetBinLowEdge(1), 1, ratio.GetXaxis().GetBinLowEdge(ratio.GetNbinsX()+1), 1)
    line.SetLineStyle(3)
    line.SetLineColor(ROOT.kBlack)
    
    for ext in ['png','pdf','root']:
        c.SaveAs("{plotdir}/{var}.{ext}".format(plotdir=plotdir,var=histo_sr.GetName(),ext=ext))
    
    
def drawSpectra(histos,plotdir):
    variables = [var for (reg,var) in histos.keys() if reg=='fe']
    print "variables to plot: ", variables

    for var in variables:
        histos[('cosm',var)].SetFillColor(ROOT.kAzure+6)
        histos[('fe',var)].SetMarkerStyle(ROOT.kFullDotLarge)
        histos[('cosm',var)].Scale(histos[('fe',var)].Integral()/histos[('cosm',var)].Integral())
        drawOne(histos[('fe',var)],histos[('cosm',var)],plotdir)

def plotEnergyVsDistance(plotdir):

    tf_fe55 = ROOT.TFile('runs/reco_run01740_3D.root')
    tree = tf_fe55.Get('Events')

    np = 9 # number of points
    dist = 100 # distance from center in pixels
    x = [dist*i for i in range(np+1)]
    y_mean = []; y_res = []

    cut_base = 'cl_iteration==2'
    integral = ROOT.TH1F('integral','',100,600,3000)
    
    for p in range(np):
        cut = "{base} && TMath::Hypot(cl_xmean-1024,cl_ymean-1024)>{r_min} && TMath::Hypot(cl_xmean-1024,cl_ymean-1024)<{r_max}".format(base=cut_base,r_min=x[p],r_max=x[p+1])
        tree.Draw('cl_integral>>integral',cut)
        mean = integral.GetMean()
        rms  = integral.GetRMS()
        y_mean.append(mean)
        y_res.append(rms/mean)
        integral.Reset()

    gr_mean = ROOT.TGraph(np,array('f',x),array('f',y_mean))
    gr_res = ROOT.TGraph(np,array('f',x),array('f',y_res))
    gr_mean.SetMarkerStyle(ROOT.kOpenCircle)
    gr_res.SetMarkerStyle(ROOT.kOpenCircle)
    gr_mean.SetMarkerSize(2)
    gr_res.SetMarkerSize(2)
    
    gr_mean.SetTitle('')
    gr_res.SetTitle('')
    
    c = ROOT.TCanvas('c','',1200,1200)
    lMargin = 0.12
    rMargin = 0.05
    bMargin = 0.30
    tMargin = 0.07

    gr_mean.Draw('AP')
    gr_mean.GetXaxis().SetTitle('distance from center (pixels)')
    gr_mean.GetYaxis().SetTitle('integral (photons)')

    for ext in ['png','pdf']:
        c.SaveAs("{plotdir}/mean.{ext}".format(plotdir=plotdir,ext=ext))

    gr_res.Draw('AP')
    gr_res.GetXaxis().SetTitle('distance from center (pixels)')
    gr_res.GetYaxis().SetTitle('resolution (rms)')
    gr_res.GetYaxis().SetRangeUser(0.10,0.50)

    for ext in ['png','pdf']:
        c.SaveAs("{plotdir}/rms.{ext}".format(plotdir=plotdir,ext=ext))

    print x
    print y_mean
    print y_res
    
    
def plotPMTEnergyVsPosition(plotdir):
    tf_fe55 = ROOT.TFile('runs/reco_run01754_to_run01759.root')
    tree = tf_fe55.Get('Events')

    runs = range(1754,1760)
    integral = ROOT.TH1F('integral','',100,2000,20000)

    np = len(runs)
    x = []
    y_mean = []; y_res = []
    
    for i,r in enumerate(runs):
        cut = 'run=={r} && pmt_tot<100'.format(r=r)
        tree.Draw('pmt_integral>>integral',cut)
        mean = integral.GetMean()
        rms  = integral.GetRMS()
        x.append(i)
        y_mean.append(mean)
        y_res.append(rms/mean)
        integral.Reset()


    print y_res
    
    gr_mean = ROOT.TGraph(np,array('f',x),array('f',y_mean))
    gr_res = ROOT.TGraph(np,array('f',x),array('f',y_res))
    gr_mean.GetXaxis().SetRangeUser(-1,np)
    gr_res.GetXaxis().SetRangeUser(-1,np)
    gr_mean.GetYaxis().SetRangeUser(5000,9000)
    gr_res. GetYaxis().SetRangeUser(0,1.0)
    gr_mean.SetMarkerStyle(ROOT.kOpenCircle)
    gr_res.SetMarkerStyle(ROOT.kOpenCircle)
    gr_mean.SetMarkerSize(2)
    gr_res.SetMarkerSize(2)
    
    gr_mean.SetTitle('')
    gr_res.SetTitle('')

    c = ROOT.TCanvas('c','',1200,1200)
    lMargin = 0.17
    rMargin = 0.05
    bMargin = 0.15
    tMargin = 0.07
    c.SetLeftMargin(lMargin)
    c.SetRightMargin(rMargin)
    c.SetTopMargin(tMargin)
    c.SetBottomMargin(bMargin)
    c.SetFrameBorderMode(0);
    c.SetBorderMode(0);
    c.SetBorderSize(0);

    gr_mean.Draw('AP')
    gr_mean.GetXaxis().SetTitle('source position index')
    gr_mean.GetYaxis().SetTitle('PMT integral (mV)')

    for ext in ['png','pdf']:
        c.SaveAs("{plotdir}/mean.{ext}".format(plotdir=plotdir,ext=ext))

    gr_res.Draw('AP')
    gr_res.GetXaxis().SetTitle('source position index')
    gr_res.GetYaxis().SetTitle('resolution (rms)')

    for ext in ['png','pdf']:
        c.SaveAs("{plotdir}/rms.{ext}".format(plotdir=plotdir,ext=ext))

    print x
    print y_mean
    print y_res


def plotCameraPMTCorr(outdir):
    tf_fe55 = ROOT.TFile('runs/reco_run01754_to_run01759.root')
    tree = tf_fe55.Get('Events')

    tot_vs_nhits = ROOT.TH2F('nhits_vs_tot','',45,20,100,30,100,400)
    tot_vs_nhits.GetXaxis().SetTitle("T.o.T. (ms)")
    tot_vs_nhits.GetYaxis().SetTitle("supercluster pixels")
    tot_vs_nhits.SetContour(100)
    
    ## fill the 2D histogram
    for event in tree:
        if event.pmt_tot > 100: continue
        for isc in range(event.nSc):
            if event.sc_iteration[isc]!=2: continue
            if event.sc_width[isc]/event.sc_length[isc]<0.7: continue
            if not withinFC(event.sc_xmean[isc],event.sc_ymean[isc],700,700): continue
            tot_vs_nhits.Fill(event.pmt_tot,event.sc_nhits[isc])

    ## profile for better understanding
    profX = tot_vs_nhits.ProfileX()
    profX.SetMarkerStyle(ROOT.kFullCircle)
    profX.SetLineColor(ROOT.kBlack)
    profX.SetMarkerColor(ROOT.kBlack)
    profX.GetYaxis().SetRangeUser(180,310)
    profX.GetYaxis().SetTitle("average pixels in SC")

    ROOT.gStyle.SetPalette(ROOT.kRainBow)
    ROOT.gStyle.SetOptStat(0)
    c = getCanvas()
    tot_vs_nhits.Draw("colz")
    for ext in ['png','pdf','root']:
        c.SaveAs('{plotdir}/{name}.{ext}'.format(plotdir=outdir,name=tot_vs_nhits.GetName(),ext=ext))

    profX.Draw('pe1')
    for ext in ['png','pdf','root']:
        c.SaveAs('{plotdir}/{name}_profX.{ext}'.format(plotdir=outdir,name=tot_vs_nhits.GetName(),ext=ext))
    
    
if __name__ == "__main__":

    parser = optparse.OptionParser(usage='usage: %prog [opts] ', version='%prog 1.0')
    parser.add_option('', '--make'   , type='string'       , default='tworuns' , help='run simple plots (options=tworuns,evsdist,pmtvsz,cluvspmt,multiplicity)')
    parser.add_option('', '--outdir' , type='string'       , default='./'      , help='output directory with directory structure and plots')
    (options, args) = parser.parse_args()

    ## make the output directory first
    os.system('mkdir -p {od}'.format(od=options.outdir))
    
    if options.make in ['all','multiplicity']:
        plotNClusters()

    if options.make in ['all','tworuns']:
        histograms = fillSpectra()
        odir = options.outdir+'/clusters/'
        os.system('mkdir -p {od}'.format(od=odir))
        drawSpectra(histograms,odir)
        os.system('cp index.php {od}'.format(od=odir))
    
    if options.make in ['all','evsdist']:
        plotEnergyVsDistance(options.outdir)

    if options.make in ['all','pmtvsz']:
        plotPMTEnergyVsPosition(options.outdir)

    if options.make in ['all','cluvspmt']:
        plotCameraPMTCorr(options.outdir)