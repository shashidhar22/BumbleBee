import vcf
import pysam
import os
import sys
import csv
import argparse
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import numpy.ma as ma
import seaborn as sns
from pyfasta import Fasta
import subprocess
from multiprocessing import Pool
#from pathos.multiprocessing import ProcessingPool as Pool
import glob
import scipy.stats
from itertools import repeat
from itertools import chain
from pprint import pprint
from sklearn.naive_bayes import BernoulliNB
from sklearn import cross_validation
from rpy2.robjects.packages import importr
import math
import operator
from rpy2.robjects import pandas2ri
pandas2ri.activate()

def create_matrix(arguments):
    dist_dict = dict()
    varfile = vcf.Reader(open(arguments[0]))
    samfile = pysam.AlignmentFile(arguments[1]) 
    fasta = Fasta(arguments[2])
    window = arguments[3]
    mincount = arguments[4]
    metdict = dict() 
    out_file = list()
    meth_file= list()
    record_array = list()
    var_accuracy = dict()
    for var in varfile:
        seen_id = set()
        if not var.genotype(var.samples[0].sample).is_het or not var.is_snp or len(var.ALT) > 1: #or (var.REF == 'C' and 'T' in var.ALT):
            continue
        record_array = list()
        index_list = list()
        context_dict = dict()
        if 'chr' not in var.CHROM:
            var.CHROM = 'chr'+var.CHROM
        for depth,reads in enumerate(samfile.fetch(region=var.CHROM+':'+str(var.POS)+':'+str(var.POS))):
            bitseq = dict()
            index = ''
            if reads.is_unmapped or reads.mate_is_unmapped:
                continue
            if reads.is_proper_pair and reads.query_name not in seen_id and 'I' not in reads.cigarstring and 'D' not in reads.cigarstring :
                seen_id.add(reads.query_name)
                mates = samfile.mate(reads)
                wat,label  = watson(reads,var.POS,var.CHROM,fasta,var.REF,str(var.ALT[0]))
                bitseq.update(wat)
                if label != '': index = label
                wat,label = watson(mates,var.POS,var.CHROM,fasta,var.REF,str(var.ALT[0]))
                bitseq.update(wat)
                if label != '': index =label
                record_array.append(bitseq)
                index_list.append(index)
            elif reads.query_name not in seen_id and 'I' not in reads.cigarstring and 'D' not in reads.cigarstring:
                seen_id.add(reads.query_name)
                wat,label  = watson(reads,var.POS,var.CHROM,fasta,var.REF,str(var.ALT[0]))
                bitseq.update(wat)
                if label != '': index = label
                record_array.append(bitseq)
                index_list.append(index)
            if depth > 1000:
                break
        if index_list.count('R') >= mincount  and index_list.count('A') >= mincount :
            dist_dict[var.CHROM+'_'+str(var.POS)] = [(index_list.count('R')*100/float(len(index_list))),(index_list.count('A')*100/float(len(index_list)))]
            records_data = pd.DataFrame(record_array,index=index_list).drop(labels='',axis=0)
            records_data.sort(axis=1).replace(4,np.nan).to_csv(str(var.CHROM)+'_'+str(var.POS)+'.csv',na_rep='NaN')
            accuracy = 'NA' #NaiveBayes(str(var.CHROM)+'_'+str(var.POS)+'.csv',str(var.CHROM)+'_'+str(var.POS))
            #os.remove(str(var.CHROM)+'_'+str(var.POS)+'.csv')
            fisher_values, fisher_tables = pvector(records_data)
            context_dict = get_met(var.CHROM,list(records_data.columns.values),fasta)
            for key,val in fisher_values.iteritems():
                if context_dict[int(key)] != 'NA':
                    meth_file.append([str(var.CHROM),str(var.POS),str(key),str(val),str(context_dict[int(key)])])
            window_frame = {key:val for key,val in fisher_values.iteritems() if context_dict[int(key)] != 'NA'}
            test_count = 1
            recursive_fishers(window_frame)
            feature_mean,window = recursive_fisher(window_frame,window,var.POS,window,test_count)
            var_accuracy[(var.CHROM)+':'+str(var.POS)] = accuracy
            rank_frame  = {key:val for key,val in window_frame.iteritems() if val <= 0.05}
            rank_fisher, sites,median = ranked_fisher(window_frame,var.POS)
            #plot_matrix(records_data,var.CHROM+'_'+str(var.POS))
            out_file.append([var.CHROM,str(var.POS),var.REF,','.join([str(vals) for vals in var.ALT]),str(len(record_array)),str(index_list.count('R')),str(index_list.count('A')),str(context_dict.values().count('CG')),str(context_dict.values().count('CHG')), str(context_dict.values().count('CHH')),str(context_dict.values().count('NA')),str(accuracy),str(len(window_frame)),str(feature_mean),str(window),str(rank_fisher),str(sites),str(median)])
            #plot_ratios(records_data,var.CHROM,var.POS,context_dict)
    mean  = np.mean(var_accuracy.values())
    std = np.std(var_accuracy.values())
    return(out_file,meth_file)

def plot_ratios(records_data,chr,pos,context_dict):
    methylation_records,context_records = counter(records_data,context_dict)
    methylation_records.to_csv(str(chr)+'_'+str(pos)+'_methratio.csv')
    plt.figure(figsize=(36, 12), facecolor='w', edgecolor='k')
    methylation_records.transpose().plot(kind='hist',alpha=0.5,bins=20)
    plt.title(str(chr)+'_'+str(pos))
    plt.ylabel('No of CpG,CHG,CHH')
    plt.xlabel('Methylation ratio')
    plt.savefig(str(chr)+'_'+str(pos)+'_methratio.png')
    plt.close()
    plt.figure(figsize=(36, 12), facecolor='w', edgecolor='k')
    context_records.transpose().plot(kind='bar')
    plt.title(str(chr)+'_'+str(pos))
    plt.savefig(str(chr)+'_'+str(pos)+'_methratio_context.png')
    plt.close()
    return

def fisher_method(fishers):
    sl = -2 *sum([math.log(vals) for vals in fishers])
    return(scipy.stats.chi2.sf(sl,df=2*len(fishers)))

def ranked_fisher(fishers,pos):
    fisher_list = list()
    distance_list = list()
    sorted_fisher = sorted(fishers.items(), key=operator.itemgetter(1))
    fmpval = 0.0
    cycle = 0
    range = ''
    for cycle,pval in enumerate(sorted_fisher,start=1):
        fisher_list.append(pval[1])
        #fmpval = fisher_method(fisher_list)
        fmpval = scipy.stats.combine_pvalues(fisher_list)[1]
        if fmpval > 0.05/cycle :
            break
        distance_list.append(int(pval[0]))
    if len(distance_list) == 0 :
        range = '0,0'
    else:
        range = str(min(distance_list)) +','+ str(max(distance_list))
    return (fmpval,cycle,range)

def recursive_fishers(fishers):
    event = dict()
    locus = dict()
    loci = list()
    pval = list()
    pos = sorted(fishers.keys())
    print(pos)
    for count,index in enumerate(pos,start=1):
        if count % 10 == 0:
            pval.append(fishers[index])
            loci.append(index)
            event[count] = fisher_method(pval)
            locus[count] = loci
            pval = list()
            loci = list()
        else:
            loci.append(index)
            pval.append(fishers[index])
    print(event)
    print(locus)
    sys.exit()
    return

def recursive_fisher(fishers,window,pos,scan,test_count):
    if pos- (window/2) in [int(vals) for vals in fishers.keys()] and (pos + window/2) in [int(vals) for vals in fishers.keys()]  and fisher_method([fishers[pos-vals]  for vals in range(-(window/2),(window/2)+1) if (pos-vals) in fishers]) <= 0.05/float(window):
        window += scan
        test_count +=1
        pval,window = recursive_fisher(fishers,window,pos,scan,test_count)
    elif pos - ((window+scan)/2) in  [int(vals) for vals in fishers.keys()]  and (pos + (window+scan)/2) in [int(vals) for vals in fishers.keys()] and  fisher_method([fishers[(pos-vals)] for vals in range(-(window/2),(window/2)+1) if (pos-vals) in fishers]) > 0.05/float(window) and fisher_method([fishers[(pos-vals)] for vals in range(-((window+scan)/2),((window+scan)/2)+1) if (pos-vals) in fishers]) <= 0.05/float(window+scan) and window+scan < len(fishers):
        window += scan
        test_count += 1
        pval, window = recursive_fisher(fishers,window,pos,scan,test_count)
    else:
        pval = fisher_method([fishers[(pos-vals)]  for vals in range(-(window/2),(window/2)+1) if (pos-vals) in fishers])
    return(pval,window)

def watson(reads, variant,chrom,fasta,snp,alt):
    positions = reads.get_aligned_pairs()
    sequence = reads.query_sequence
    meth_status = dict()
    #meth_context = dict()
    pos =0 
    label = ''
    while pos < len(positions):
        frag, ref = positions[pos]
        if ref != None and frag != None:
            if ref == variant:
                label = label_maker(reads,frag,sequence,snp,alt)
            if sequence[frag].upper() == 'C' and str(fasta[chrom][ref]).upper() == 'C':
                meth_status[ref+1] = 0
            elif sequence[frag].upper() == 'T' and str(fasta[chrom][ref]).upper() == 'C':
                meth_status[ref+1] = 2
            else:
                meth_status[ref+1] = 4
        elif frag == None:
            meth_status[ref+1] = np.nan
        pos += 1
    return(meth_status,label)

def label_maker(reads,frag,sequence,snp,alt):
    antisense = {'CTC':'R','CTT':'A','CGC':'R','CGG':'A','CGA':'A','CAC':'R','CAA':'A','ATA':'R','ATT':'A','ACA':'R','ACC':'A','GCG':'R','GCA':'R',
                 'GCC':'A','GTG':'R','GTA':'R','GTT':'A','TAT':'R','TAA':'A','TCT':'R','TCC':'A','TGT':'R','TGG':'A','TGA':'A'}
    sense = {'CGC':'R','CGT':'R','CGG':'A','CAC':'R','CAT':'R','CAA':'A','ATA':'R','ATT':'A','ACA':'R','ACC':'A','ACT':'A','AGA':'R',
             'AGG':'A','GAG':'R','GAA':'A','GCG':'R','GCC':'A','GCT':'A','GTG':'R','GTT':'A','TAT':'R','TAA':'A','TGT':'R','TGG':'A'}
    if ((snp == 'C' and alt == 'T') or (snp == 'T' and alt == 'C')) and not (reads.flag & 0x10):
        return ('')
    elif ((snp == 'G' and alt == 'A') or (snp == 'A' and alt == 'G')) and reads.flag & 0x10:
        return ('')
    elif reads.flag & 0x10:
        try:
            return(antisense[snp+alt+sequence[frag-1]])
        except KeyError:
            return('')
    else:
        try:
            return(sense[snp+alt+sequence[frag-1]])
        except KeyError:
            return('')

def get_met(chrom,positions,fasta):
    context = dict()
    for count,base in enumerate(positions):
        if str(fasta[chrom][base-1]).upper() == 'C':
            if str(fasta[chrom][base]).upper() == 'G':
                context[base] ='CG'
            elif str(fasta[chrom][base+1]).upper() == 'G':
                context[base] = 'CHG'
            else:
                context[base] = 'CHH'
        elif str(fasta[chrom][base-1]).upper() == 'G':
            if str(fasta[chrom][base-2]).upper() == 'C':
                context[base] = 'CG'
            elif str(fasta[chrom][base-3]).upper() == 'C':
                context[base] = 'NA'
            else:
                context[base] = 'NA'
        else:
            context[base] = 'NA'
    return(context)

def modifier(cigar):
    mod = 0 
    for vals in cigar:
        if vals[0] == 1:
            mod += vals[1]
        elif vals[0] == 2:
            mod += vals[1]
    return(mod)
def plot_matrix(records_data,name):
    records_data = records_data.sort().sort(axis=1).fillna(6)
    plt.figure(figsize=(36, 12), facecolor='w', edgecolor='k')
    palate=sns.color_palette("Set1", n_colors=8, desat=.5)
    sns.heatmap(records_data,square=False,cmap='gist_stern')
    plt.xticks(rotation=90)
    plt.savefig(name)
    plt.close()
    return

def fisher_test(values,labels):
    cont = np.zeros((2,2))
    for met,snp in zip(values,labels):
        if met == 0 and snp == 'R':
            cont[0,0] += 1
        elif met == 0 and snp == 'A':
            cont[0,1] += 1
        elif met == 2 and snp =='R':
            cont[1,0] += 1
        elif met == 2 and snp == 'A':
            cont[1,1] += 1
        else:
            continue
    oddsratio, pvalue = scipy.stats.fisher_exact(cont)
    return(pvalue,cont)

def pvector(records_data):
    feature_frame = records_data.transpose()
    labels = list(feature_frame.columns.values)
    sig_vector = dict()
    cont_vector = dict()
    for row,values in feature_frame.iterrows():
        sig_vector[int(row)],cont_vector[int(row)] = fisher_test(values,labels)
    return(sig_vector,cont_vector)

def counter(records_data,context_dict):
    count_list = dict()
    context_list = dict()
    for column,series in records_data.iteritems():
        mref,uref,malt,ualt = 0,0,0,0
        for indices,values in series.iteritems():
            if values == 0.0 and indices == 'R':
                mref +=1
            elif values == 2.0 and indices == 'R':
                uref += 1
            elif values ==0.0  and indices == 'A':
                malt += 1
            elif values == 2.0 and indices == 'A':
                ualt += 1
        try:
            met_ref = mref/float(uref+mref)
        except ZeroDivisionError:
            met_ref = np.nan
        try:
            met_alt = malt/float(ualt+malt)
        except ZeroDivisionError:
            met_alt = np.nan
        count_list[column] = pd.Series([met_ref,met_alt], index=['Ref','Alt'])
        context_list[context_dict[column]] = pd.Series([met_ref,met_alt],index=['Ref','Alt'])
    return(pd.DataFrame(count_list),pd.DataFrame(context_list))

def NaiveBayes(records_data):
    eozso = importr('e1071')
    records = com.convert_to_r_dataframe(records_data)
    records[records == 2] =1


def NaiveBayes(records_data,var):
    run_asm = subprocess.Popen(['/nfs/sw/R/R-3.2.1/bin/Rscript','/nethome/sravishankar/projects/AlleleSpecificMethylation/asmdetector.R',records_data],stdout=subprocess.PIPE, stdin=subprocess.PIPE,shell=False)
    run_res = run_asm.communicate()
    try:
        accuracy = float(run_res[0].split('\n')[1].split(' ')[1])
    except IndexError:
        accuracy = np.nan
    return(accuracy)

def split_chrom(spl):
    inp = vcf.Reader(open(spl[0]))
    opt = vcf.Writer(open(spl[1]+'.vcf','w'),inp)
    try:
        for lines in inp.fetch(spl[1],0,10000000000):
            opt.write_record(lines)
    except ValueError:
        print(spl[1])
    return (spl[1]+'.vcf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detecting allele specific methylation')
    parser.add_argument('-v','--vcffile',type=str,help='Path to VCF file.')
    parser.add_argument('-b','--bam',type=str,help='Path to BAM/SAM file.')
    parser.add_argument('-f','--fasta',type=str,help='Reference fasta file.')
    parser.add_argument('-t','--threads',type=int,help='Number of threads.')
    parser.add_argument('-w','--windowsize',type=int,help='WIndow size for DMR detection.')
    parser.add_argument('-d','--mindepth',type=int,help='Minimum depth per allele.')
    parser.add_argument('-c','--chromannotated',action='store_true',help='VCF has chr prefix',default=False)
    parser.add_argument('-o','--outfile',type=str,help='Output file path')
    args = parser.parse_args()
    p = Pool(args.threads)
    vcffiles = args.vcffile
    bamfile = args.bam
    fastafile = args.fasta
    window = args.windowsize
    chr = args.chromannotated
    if chr:
        chrm_map = chain(['chr'+str(vals) for vals in range(1,23)],['chrX','chrY'])
    else:
        chrm_map = chain([str(vals) for vals in range(1,23)],['X','Y'])
    vcf_list = p.map(split_chrom,zip(repeat(vcffiles),chrm_map)) #[split_chrom(vcffiles,vals) for vals in chrm_map]
    #sys.exit()
    out_file = csv.writer(open(args.outfile+'.tsv','w'),delimiter='\t')
    meth_file = csv.writer(open(args.outfile+'_meth.tsv','w'),delimiter='\t')
    #distribution = p.map(create_matrix, zip(vcf_list,repeat(bamfile),repeat(fastafile),repeat(window),repeat(args.mindepth)))
    distribution = create_matrix([vcffiles,bamfile,fastafile,window,args.mindepth])
    out_file.writerow(['Chrom','Pos','Ref','Alt','Depth','Ref Count','Alt Count','No of CpGs','No of CHG','No of CHH','No of NA','Naive Bayes Accuracy','No of significant features','Recursive Fisher\'s method','Window Considered','Ranked Fisher\'s','Sites considered','Median distance from SNP'])
    meth_file.writerow(['Chrom','SNPPos','MethylationPos','pval','Context'])
    for sections in distribution:
        for lines in sections[0]:
            out_file.writerow(lines)
    for sections in distribution:
        for lines in sections[1]:
            meth_file.writerow(lines)
    for files in vcf_list:
        os.remove(files)
