from cmath import nan
import os.path
import os
import copy
from unittest import skip, skipUnless
import pandas as pd
import numpy as np
from statistics import mean, median, stdev
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html     https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html
from scipy.stats import poisson, norm
import scipy
import numpy as np
# mluvit o statistice a proc jsem ji pouzil
# implementace + testování ověření
#user acceptance testing (rozhovor s Pavlou) # nacitani csv -> UTF8 - problem proto encoding


def odhad_LoD(LoB, sm_odch, alfa):
    kde_mean = []
    if sm_odch == 0:
        return 0
    else:
        for i in np.arange(LoB+sm_odch, LoB+sm_odch*2.5, 0.001):
            kde_mean.append(i)
        mu = np.nanmean(kde_mean)
        sigma = 1 - alfa
        a = norm.ppf(sigma, kde_mean, sm_odch)
        s = np.argmin(np.abs(a-LoB))

    LoD = kde_mean[s]
    return LoD


def matrixCreator_csv(xlsx_csv, countlist, matrixfile):
    xlsx_csv = int(xlsx_csv)
    countlist = int(countlist)
    U = []
    V = []
    dolni_limit = []
    horni_limit = []
    mutace_A = []
    mutace_C = []
    mutace_G = []
    mutace_T = []
    zbytek = []
    LoD_gen = []
    Tranzice_LoD_pozice = []
    Transverze_LoD_pozice = []
    files = [os.path.join(matrixfile, x) for x in os.listdir(matrixfile) if x.endswith(
        ".csv")]  # ziskam cely obsah slozky na CSV soubory s celou cestou
    filenames = [x for x in os.listdir(matrixfile) if x.endswith(".csv")]
    folder_toSave = os.path.join(matrixfile+"Results/")
    if not os.path.exists(matrixfile+"Results"):
        os.makedirs(matrixfile+"Results")
    if countlist == 0:
        countlist = len(filenames)
    for i in range(countlist):
        T = pd.read_csv(files[i], encoding='latin-1', sep=";") # UTF8 - problem proto encoding
        U.append(T.iloc[4:, 17:21])
        V.append(T.iloc[4:, 7:15])
        pozice = T.iloc[4:, 5]

    table = pd.read_csv(files[-1], encoding='latin-1', sep=";")
    tableColumnsNames = table.iloc[3, :]
    table.columns = tableColumnsNames
    table = table.iloc[3:, :]
    gen = table.iloc[:, 2]
    Pozice_zamena = pd.read_excel("static/defaultCSV/pozice_vse.xlsx")

    # zdravi slabi - list 2-11
    # nemocni slabi - list 12-19
    alfa = 0.95
    # HK=0.75; #aby byla vy?azena min. 1 hodnota
    # hledani minimalniho poctu readu
    forward = 1000  # pro dolni_limit
    backward = 1000  # pro horni_limit
    try:
        for i in range(len(U)):
            normala = V[i].astype("float64")
            Forw = normala.iloc[:, 0::2]
            Forw = Forw.sum(axis=1)
            Backw = normala.iloc[:, 1::2]
            Backw = Backw.sum(axis=1)
            min_pocet = np.minimum(Forw, Backw)
            n = len(min_pocet)
            # ted jak urcit ty hranicni body???
            dolni_limit.append(
                max(max(np.where(((min_pocet[0:round(n/2)] < forward)))))+1)
            hornifind = [
                min(min(np.where(((min_pocet[dolni_limit[i]+10:-1] < backward)))))]
            if hornifind is None:
                horni_limit.append(1433)
            else:
                horni_limit.append(hornifind+dolni_limit[i]+8+1)
        #  neni tam chyba?? matrixCreator_csv.m ->vzdycky se bere jen ten nejvetsi v programu??
        if len(horni_limit) < 2:
            horni_limit = horni_limit[0][0]
        else:
            horni_limit = min(min(horni_limit))
        dolni_limit = max(dolni_limit)

        # min1000=[];
        seznam = ['A', 'C', 'G', 'T']
        geny = np.zeros(len(gen)-1)
        for g in range(len(seznam)):
            for i in range(1, len(gen)):
                if gen.iloc[i] == seznam[g]:
                    geny[i-1] = g+1
        # spatne=[1:dolni_limit-1 horni_limit+1:size(geny,1)]'; #geny, kde jich je mene nez 1000 (rucne hledano, potreba zautomatizovat)
        # geny(spatne,:)=[];
        # pozice(spatne,:)=[];
        # U_vse=U;
        # U=cellfun(@(x) x(setdiff(1:size(U_vse{1},1),spatne),:),U_vse,'un',false);

        A = geny == 1
        C = geny == 2
        G = geny == 3
        T = geny == 4

        for i in range(len(U)):
            mutace_A.append([[U[i].iloc[:, [1, 2, 3]][A == True].astype("Float64").mean()], [
                            U[i].iloc[:, [1, 2, 3]][A == True].astype("Float64").std()]])
            mutace_C.append([[U[i].iloc[:, [0, 2, 3]][C == True].astype("Float64").mean()], [
                            U[i].iloc[:, [0, 2, 3]][C == True].astype("Float64").mean()]])
            mutace_G.append([[U[i].iloc[:, [0, 1, 3]][G == True].astype("Float64").mean()], [
                            U[i].iloc[:, [0, 1, 3]][G == True].astype("Float64").mean()]])
            mutace_T.append([[U[i].iloc[:, [0, 1, 2]][T == True].astype("Float64").mean()], [
                            U[i].iloc[:, [0, 1, 2]][T == True].astype("Float64").mean()]])

        # kontroly - zdravi
        sumzdrav = np.zeros((len(U[0]), len(U[0].iloc[0])))
        for i in range(len(U)):
            sumzdrav += U[i].astype("Float64")
        zdrav = sumzdrav[dolni_limit:horni_limit+1].astype("Float64")
        suma_genu = zdrav.sum(axis=1)
        A = A[dolni_limit:horni_limit+1]
        C = C[dolni_limit:horni_limit+1]
        G = G[dolni_limit:horni_limit+1]
        T = T[dolni_limit:horni_limit+1]

        # predzpracovani dat
        x = range(0, 33, 1)
        snip = [[441, 2, 3], [479, 2, 0], [543, 0, 2], [
            804, 0, 2], [912, 0, 2], [951, 3, 1], [1689, 2, 0]]
        for i in range(len(U)):
            # (max(max(np.where(((min_pocet[0:round(n/2)]<forward)))))+1)
            for j in range(len(snip)):
                # np.where((pozice.iloc[:].astype("Float64") ==snip[0][0], U[0].iloc[:,0].astype("Float64") ==snip[0][1]))
                SNPVPozice = np.where(
                    pozice.iloc[:].astype("Float64") == snip[j][0])
                SNPV = float(U[i].iloc[SNPVPozice[0][0], snip[j][1]])
                SNPS = float(U[i].iloc[SNPVPozice[0][0], snip[j][2]])
                if SNPV/SNPS > 0.2:
                    U[i].iloc[SNPVPozice[0][0], snip[j][2]] = SNPV + SNPS
                    U[i].iloc[SNPVPozice[0][0], snip[j][1]] = 0
        # for i=1:size(poc_tranzice_norm,1)
        #    At = poc_tranzice_norm(i,:);
        #    if mean(At) > 100
        #         [TFt,Lt,Ut,Ct] = isoutlier(At, 'percentiles', [0,88]);
        #         plot(x,At,x(TFt),At(TFt),'x',x,Lt*ones(1,34),x,Ut*ones(1,34),x,Ct*ones(1,34))
        #         legend('Original Data','Outlier','Lower Threshold','Upper Threshold','Center Value')
        #         [B, TF2] = rmoutliers(poc_tranzice_norm(i,:), 'percentiles', [0,88]);
        #         plot(x,At,'b.-',x(~TF2),B,'r-')
        #         legend('Input Data','Output Data')
        #         At(TF2)= NaN;
        #         AFM = fillmissing(At, 'linear');
        #         poc_tranzice_norm(i,:)= AFM;
        #    end
        # end
        # for i=1:size(poc_transverze_norm,1)
        #    At = poc_transverze_norm(i,:);
        #    if mean(At) > 100
        #         [TFt,Lt,Ut,Ct] = isoutlier(At, 'percentiles', [0,88]);
        #         plot(x,At,x(TFt),At(TFt),'x',x,Lt*ones(1,34),x,Ut*ones(1,34),x,Ct*ones(1,34))
        #         legend('Original Data','Outlier','Lower Threshold','Upper Threshold','Center Value')
        #         [B, TF2] = rmoutliers(poc_transverze_norm(i,:), 'percentiles', [0,88]);
        #         plot(x,At,'b.-',x(~TF2),B,'r-')
        #         legend('Input Data','Output Data')
        #         At(TF2)= NaN;
        #         AFM = fillmissing(At, 'linear');
        #         poc_transverze_norm(i,:)= AFM;
        #    end

        # LoB po genech
        # AG
        AG_celkem = zdrav.iloc[:, [2]][A == True].astype(
            "Float64")  # vybere vse, kde ma byt A a je tam G
        AG_1vzorek = zdrav.iloc[:, [2]][A == True].astype("Float64") / len(U)
        norm_AG = AG_1vzorek.iloc[:, 0] * np.divide(
            3000*len(U), suma_genu.iloc[:][A == True].astype("Float64"))
        AG_stat = [AG_1vzorek.iloc[:].sum(), AG_1vzorek.iloc[:].mean(
        ), AG_1vzorek.iloc[:].std(), AG_1vzorek.iloc[:].median()]
        norm_AG_stat = [norm_AG.iloc[:].sum(), norm_AG.iloc[:].mean(
        ), norm_AG.iloc[:].std(), norm_AG.iloc[:].median()]
        # Fit the data to a Poisson distribution
        mu = np.mean(norm_AG)
        # mean, var, skew, kurt = poisson.stats(mu, moments='mvsk')
        var = poisson.stats(mu, moments='mvsk')
        AG_L = var[0]
        AG_k = [AG_L, poisson.ppf(alfa, var[0])]
        # GA
        GA_celkem = zdrav.iloc[:, [0]][G == True].astype(
            "Float64")  # vybere vse, kde ma byt G a je tam A
        GA_1vzorek = zdrav.iloc[:, [0]][G == True].astype("Float64") / len(U)
        norm_GA = GA_1vzorek.iloc[:, 0] * np.divide(
            3000*len(U), suma_genu.iloc[:][G == True].astype("Float64"))
        GA_stat = [GA_1vzorek.iloc[:].sum(), GA_1vzorek.iloc[:].mean(
        ), GA_1vzorek.iloc[:].std(), GA_1vzorek.iloc[:].median()]
        norm_GA_stat = [norm_GA.iloc[:].sum(), norm_GA.iloc[:].mean(
        ), norm_GA.iloc[:].std(), norm_GA.iloc[:].median()]
        # Fit the data to a Poisson distribution
        mu = np.mean(norm_GA)
        # mean, var, skew, kurt = poisson.stats(mu, moments='mvsk')
        var = poisson.stats(mu, moments='mvsk')
        GA_L = var[0]
        GA_k = [GA_L, poisson.ppf(alfa, var[0])]
        # TC
        TC_celkem = zdrav.iloc[:, [1]][T == True].astype(
            "Float64")  # vybere vse, kde ma byt T a je tam C
        TC_1vzorek = zdrav.iloc[:, [1]][T == True].astype("Float64") / len(U)
        norm_TC = TC_1vzorek.iloc[:, 0] * np.divide(
            3000*len(U), suma_genu.iloc[:][T == True].astype("Float64"))
        TC_stat = [TC_1vzorek.iloc[:].sum(), TC_1vzorek.iloc[:].mean(
        ), TC_1vzorek.iloc[:].std(), TC_1vzorek.iloc[:].median()]
        norm_TC_stat = [norm_TC.iloc[:].sum(), norm_TC.iloc[:].mean(
        ), norm_TC.iloc[:].std(), norm_TC.iloc[:].median()]
        # Fit the data to a Poisson distribution
        mu = np.mean(norm_TC)
        # mean, var, skew, kurt = poisson.stats(mu, moments='mvsk')
        var = poisson.stats(mu, moments='mvsk')
        TC_L = var[0]
        TC_k = [TC_L, poisson.ppf(alfa, var[0])]
        # CT
        CT_celkem = zdrav.iloc[:, [3]][C == True].astype(
            "Float64")  # vybere vse, kde ma byt G a je tam A
        CT_1vzorek = zdrav.iloc[:, [3]][C == True].astype("Float64") / len(U)
        norm_CT = CT_1vzorek.iloc[:, 0] * np.divide(
            3000*len(U), suma_genu.iloc[:][C == True].astype("Float64"))
        CT_stat = [CT_1vzorek.iloc[:].sum(), CT_1vzorek.iloc[:].mean(
        ), CT_1vzorek.iloc[:].std(), CT_1vzorek.iloc[:].median()]
        norm_CT_stat = [norm_CT.iloc[:].sum(), norm_CT.iloc[:].mean(
        ), norm_CT.iloc[:].std(), norm_CT.iloc[:].median()]
        # Fit the data to a Poisson distribution
        mu = np.mean(norm_CT)
        # mean, var, skew, kurt = poisson.stats(mu, moments='mvsk')
        var = poisson.stats(mu, moments='mvsk')
        CT_L = var[0]
        CT_k = [CT_L, poisson.ppf(alfa, var[0])]

        # pro ostatni zameny
        raz = zdrav.iloc[:, [1, 3]][A == True].astype(
            "Float64").iloc[:].sum(axis=1).iloc[:]
        dva = zdrav.iloc[:, [0, 2]][C == True].astype(
            "Float64").iloc[:].sum(axis=1).iloc[:]
        tri = zdrav.iloc[:, [1, 3]][G == True].astype(
            "Float64").iloc[:].sum(axis=1).iloc[:]
        ctiry = zdrav.iloc[:, [0, 2]][T == True].astype(
            "Float64").iloc[:].sum(axis=1).iloc[:]
        # zbytek = [*raz , *dva , *tri , *ctiry]
        zbytek = pd.DataFrame([*raz, *dva, *tri, *ctiry])
        zbytek_1vzorek = zbytek.iloc[:, 0].astype("Float64") / len(U)
        maticeNaNasobeni = np.divide(
            3000*len(U), suma_genu.iloc[:].astype("Float64"))
        norm_zbytek = np.zeros(len(zbytek_1vzorek))
        for i in range(len(zbytek_1vzorek)):
            norm_zbytek[i] = zbytek_1vzorek.iloc[i] * maticeNaNasobeni.iloc[i]
        zbytek_stat = [zbytek_1vzorek.iloc[:].sum(), zbytek_1vzorek.iloc[:].mean(
        ), zbytek_1vzorek.iloc[:].std(), zbytek_1vzorek.iloc[:].median()]
        norm_zbytek_stat = [np.sum(norm_zbytek), np.mean(
            norm_zbytek), np.std(norm_zbytek), np.median(norm_zbytek)]
        mu = np.mean(norm_zbytek)
        # mean, var, skew, kurt = poisson.stats(mu, moments='mvsk')
        var = poisson.stats(mu, moments='mvsk')
        zbytek_L = var[0]
        zbytek_k = [zbytek_L, poisson.ppf(alfa, var[0])]

        vse = zdrav.sum(axis=1) - zdrav.max(axis=1)
        vse_1vzorek = vse.iloc[:].astype("Float64") / len(U)
        maticeNaNasobeni = np.divide(
            3000*len(U), suma_genu.iloc[:].astype("Float64"))
        norm_vse = np.zeros(len(vse_1vzorek))
        for i in range(len(vse_1vzorek)):
            norm_vse[i] = vse_1vzorek.iloc[i] * maticeNaNasobeni.iloc[i]
        vse_stat = [vse_1vzorek.iloc[:].sum(), vse_1vzorek.iloc[:].mean(
        ), vse_1vzorek.iloc[:].std(), vse_1vzorek.iloc[:].median()]
        norm_vse_stat = [np.sum(norm_vse), np.mean(
            norm_vse), np.std(norm_vse), np.median(norm_vse)]
        mu = np.mean(norm_vse)
        # mean, var, skew, kurt = poisson.stats(mu, moments='mvsk')
        var = poisson.stats(mu, moments='mvsk')
        vse_L = var[0]
        vse_k = [vse_L, poisson.ppf(alfa, var[0])]

        # LoD po genech
        LoB_gen = [[AG_k[1], [norm_AG_stat[2]]],
                [GA_k[1], [norm_GA_stat[2]]],
                [TC_k[1], [norm_TC_stat[2]]],
                [CT_k[1], [norm_CT_stat[2]]],
                [zbytek_k[1], [norm_zbytek_stat[2]]]]
        for i in range(LoB_gen.__len__()):
            LoD_gen.append(odhad_LoD(LoB_gen[i][0], LoB_gen[i][1][0], alfa))

        AG_LoD = LoD_gen[0]     # G_LoD=19.4;
        GA_LoD = LoD_gen[1]     # GA_LoD=7.5;
        TC_LoD = LoD_gen[2]     # TC_LoD=20.65;
        CT_LoD = LoD_gen[3]     # CT_LoD=6;
        zbytek_LoD = LoD_gen[4]  # zbytek_LoD=4.25;

        AG_LoB = LoB_gen[0][0]
        GA_LoB = LoB_gen[1][0]
        TC_LoB = LoB_gen[2][0]
        CT_LoB = LoB_gen[3][0]
        zbytek_LoB = LoB_gen[4][0]

    # zmeny po pozicich
    # pocet zmen transverze a tranzice
        koeficient = np.zeros((len(geny), len(U)))
        poc_tranzice = np.zeros((len(geny), len(U)))
        poc_transverze = np.zeros((len(geny), len(U)))
        for i in range(len(U)):
            W = U[i]
            koeficient[:, i] = W.iloc[:].astype("Float64").sum(axis=1)/3000
            typ_zmen = [[0, 1, 2, 3], [2, 0, 3, 1], [1, 2, 0, 2], [3, 2, 1, 0]]
            for j in range(4):  
                poz = j
                kde = geny == poz+1
                # poc_tranzice.append(W.iloc[:,[typ_zmen[1][j]]].astype("Float64"))
                # poc_tranzice[-1] = poc_tranzice[-1].iloc[:,0].mul(kde)
                poc_tranzice[kde, i] = W.iloc[:, [typ_zmen[1][j]]].astype(
                    "Float64").iloc[kde, 0]
                # poc_transverze.append(W.iloc[:,[typ_zmen[2][j],typ_zmen[3][j]]].astype("Float64").sum(axis = 1))
                # poc_transverze[-1] = poc_transverze[-1].iloc[0:-1].mul(kde[0:-1])
                poc_transverze[kde, i] = W.iloc[kde, [typ_zmen[2]
                                                    [j], typ_zmen[3][j]]].astype("Float64").sum(axis=1)
        poc_tranzice_norm = np.divide(poc_tranzice, koeficient)
        poc_transverze_norm = np.divide(poc_transverze, koeficient)
            # limity pro LoD pres vsechny
        # Limit=[pozice geny zeros(size(geny,1),2)];
        Limit = [list(Pozice_zamena.pozice), Pozice_zamena.zamena, list(np.zeros(len(Pozice_zamena))), list(np.zeros(len(Pozice_zamena)))]
        limity_LoD = Limit
        siz = len(limity_LoD)
        for i in range(len(limity_LoD[0])):
            if limity_LoD[1][i] == 1:
                if limity_LoD[2][i] < AG_LoD:
                    limity_LoD[2][i] = AG_LoD
                if limity_LoD[3][i] < zbytek_LoD:
                    limity_LoD[3][i] = zbytek_LoD
            elif limity_LoD[1][i] == 2:
                if limity_LoD[2][i] < CT_LoD:
                    limity_LoD[2][i] = CT_LoD
                if limity_LoD[3][i] < zbytek_LoD:
                    limity_LoD[3][i] = zbytek_LoD
            elif limity_LoD[1][i] == 3:
                if limity_LoD[2][i] < GA_LoD:
                    limity_LoD[2][i] = GA_LoD
                if limity_LoD[3][i] < zbytek_LoD:
                    limity_LoD[3][i] = zbytek_LoD
            elif limity_LoD[1][i] == 4:
                if limity_LoD[2][i] < TC_LoD:
                    limity_LoD[2][i] = TC_LoD
                if limity_LoD[3][i] < zbytek_LoD:
                    limity_LoD[3][i] = zbytek_LoD
        # limity pro LoB, zjisti se, kde je co spatne
        Limit = [Pozice_zamena.pozice, Pozice_zamena.zamena, list(np.zeros(len(Pozice_zamena))), list(np.zeros(len(Pozice_zamena)))]
        limity_LoB = Limit
        for i in range(len(limity_LoD[0])):
            if limity_LoB[1][i] == 1:
                if limity_LoB[2][i] < AG_LoB:
                    limity_LoB[2][i] = AG_LoB
                if limity_LoB[3][i] < zbytek_LoB:
                    limity_LoB[3][i] = zbytek_LoB
            elif limity_LoB[1][i] == 2:
                if limity_LoB[2][i] < CT_LoB:
                    limity_LoB[2][i] = CT_LoB
                if limity_LoB[3][i] < zbytek_LoB:
                    limity_LoB[3][i] = zbytek_LoB
            elif limity_LoB[1][i] == 3:
                if limity_LoB[2][i] < GA_LoB:
                    limity_LoB[2][i] = GA_LoB
                if limity_LoB[3][i] < zbytek_LoB:
                    limity_LoB[3][i] = zbytek_LoB
            elif limity_LoB[1][i] == 4:
                if limity_LoB[2][i] < TC_LoB:
                    limity_LoB[2][i] = TC_LoB
                if limity_LoB[3][i] < zbytek_LoB:
                    limity_LoB[3][i] = zbytek_LoB
        norm_tranzice_stat = np.array([np.nansum(poc_tranzice_norm, axis=1), np.nanmean(poc_tranzice_norm, axis=1),
        np.nanstd(poc_tranzice_norm, axis = 1),  np.nanmedian(poc_tranzice_norm, axis=1)]) 
        
        #oprava protože hází špatný výsledky -> stdev hazi spravny vysledek
        for i in range(len(poc_tranzice_norm)):
            if norm_tranzice_stat[2][i]!=0:
                norm_tranzice_stat[2][i] = stdev(poc_tranzice_norm[i])

        norm_transverze_stat = np.array([np.nansum(poc_transverze_norm, axis=1), np.nanmean(poc_transverze_norm, axis=1),
        np.nanstd(poc_transverze_norm, axis=1), np.nanmedian(poc_transverze_norm, axis=1)])
        #oprava protože hází špatný výsledky -> stdev hazi spravny vysledek
        for i in range(len(poc_transverze_norm)):
            if norm_transverze_stat[2][i]!=0:
                norm_transverze_stat[2][i] = stdev(poc_transverze_norm[i])


        # LoB pres Poissona zdravych
        mu = np.mean(poc_transverze_norm, axis=1) #norm_transverze_stat[1]
        Transverze_LoB_pozice = [mu, poisson.ppf(alfa, mu)]
        mu = np.mean(poc_tranzice_norm, axis=1) #norm_transverze_stat[1]
        Tranzice_LoB_pozice = [mu, poisson.ppf(alfa, mu)]

        # LoD pres Poissona zdravych
        for i in range(len(Transverze_LoB_pozice[0])):

            if np.isnan(Tranzice_LoB_pozice[1][i]):
                Tranzice_LoD_pozice.append(nan)
            else:
                Tranzice_LoD_pozice.append(odhad_LoD(Tranzice_LoB_pozice[1][i],norm_tranzice_stat[2][i], alfa))
            if np.isnan(Transverze_LoB_pozice[1][i]):
                Transverze_LoD_pozice.append(nan)
            else:
                Transverze_LoD_pozice.append(odhad_LoD(Transverze_LoB_pozice[1][i],norm_transverze_stat[2][i],alfa))

        limity_poz_LoB=np.array([pozice, geny, Tranzice_LoB_pozice[1], Transverze_LoB_pozice[1]]) #limity_poz_LoB[0:4,29][0:4]
        limity_poz_LoD=np.array([pozice, geny, Tranzice_LoD_pozice, Transverze_LoD_pozice])
        komplet_limity_LoB=np.array(limity_LoB)
        zac = np.where(komplet_limity_LoB[0] == float(limity_poz_LoB[0][0]))
        kon = np.where(komplet_limity_LoB[0] == float(limity_poz_LoB[0][-1]))
        kde = []
        for i in range(2,4,1):
            kde = np.where(limity_LoB[zac[0][0]:kon[0][0]][i] < limity_poz_LoB[i])
            komplet_limity_LoB[i][kde] = limity_poz_LoB[i,kde][0]

        #misto pro graf : nedelam
        komplet_limity_LoD=np.array(limity_LoD)
        kde = []
        for i in range(2,4,1):
            kde = np.where(limity_LoD[zac[0][0]:kon[0][0]][i] < limity_poz_LoD[i])
            komplet_limity_LoD[i][kde] = limity_poz_LoD[i,kde][0]
        #misto pro graf : nedelam
        komplet_limity_LoQ= komplet_limity_LoD[0:4] * np.array([[1],[1], [5], [5]])

        # ukladani vysledku
        writer = pd.ExcelWriter(os.path.join(folder_toSave, 'AllSoTStatistic.xlsx'), engine='openpyxl')
        if xlsx_csv == 1:
            df = pd.DataFrame(limity_LoD)
            df.T.to_excel(writer, sheet_name='LoD_vse',index=False,header=False)
            df = pd.DataFrame(limity_poz_LoD) #df.iloc[0].astype("Float64")
            df.iloc[0] = df.iloc[0].astype("Float64")
            df.T.to_excel(writer, sheet_name='LoD_poz',index=False,header=False,na_rep='')
            df = pd.DataFrame(komplet_limity_LoD)
            df.T.to_excel(writer, sheet_name='LoD_komplet',index=False,header=False)
            df = pd.DataFrame(komplet_limity_LoQ)
            df.T.to_excel(writer, sheet_name='LoQ5_komplet',index=False,header=False)
            df = pd.DataFrame(komplet_limity_LoB)
            df.T.to_excel(writer, sheet_name='AllSoTLoBAll',index=False,header=False)

            df = pd.DataFrame(komplet_limity_LoD)
            df.T.to_csv(os.path.join(folder_toSave, 'SoTNextDOM.csv'), sep='\t', index=False,header=False)
            finalcsv = os.path.join(folder_toSave, 'SoTNextDOM.csv')
            writer.close()
        else:
            df = pd.DataFrame(komplet_limity_LoB)
            df.T.to_csv(os.path.join(folder_toSave, 'AllSoTLoBAll.csv'), sep='\t',index=False,header=False)
            df = pd.DataFrame(limity_LoD)
            df.T.to_csv(os.path.join(folder_toSave, 'AllSoTLoDAll.csv'), sep='\t',index=False,header=False)
            df = pd.DataFrame(limity_poz_LoD)
            df.T.to_csv(os.path.join(folder_toSave, 'AllSoTLoDPos.csv'), sep='\t',index=False,header=False, na_rep='')
            df = pd.DataFrame(komplet_limity_LoD)
            df.T.to_csv(os.path.join(folder_toSave, 'AllSoTLoDComplet.csv'), sep='\t', index=False,header=False)
            df = pd.DataFrame(komplet_limity_LoQ)
            df.T.to_csv(os.path.join(folder_toSave, 'AllSoTLoQComplet.csv'), sep='\t', index=False,header=False)


            df = pd.DataFrame(komplet_limity_LoD)
            df.T.to_csv(os.path.join(folder_toSave, 'SoTNextDOM.csv'), sep='\t', index=False,header=False)
            finalcsv = os.path.join(folder_toSave, 'SoTNextDOM.csv')
        # limity poradi
        # statistika poradi
    except Exception as e:
        error_message = "Pravděpodobně jste zadali špatný soubor/y.\nZkuste to znovu"
        raise Exception(error_message)
    return folder_toSave

#input = 0  # proměnná která rozlišuje zda je vstupem 1=XLSX nebo 0=CSV
#xlsx_csv = 0
#countlist = 0  # number of normals


#def test():
#    pass


#test()
#if input == 1:
#    pass
#else:  # cesta k folder of healthy Donors
#    matrixfile = "C:/Users/koude/OneDrive - České vysoké učení technické v Praze/FBMI_MAG/Diplomova_prace/Klempond/git/Nextdom/HTMLaCSSaFlask/Thresholds_Matlab/Tutotial Data/\HealthyDonors"
#    matrixCreator_csv(xlsx_csv, countlist, matrixfile)
