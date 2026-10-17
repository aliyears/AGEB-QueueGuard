# AGEB-QueueGuard (Team AGEB) - IBM Z Datathon 2026

Tema: Real-Time AI for Critical Decisions. Arastirma amacli hackathon prototipidir; canli banka islemleri uzerinde dogrulanmis bir guvenlik urunu degildir ve gercek islemleri engellemez.

## Fikir
Her islemi pahali modelden gecirmek yerine once hafif modelle skorlamak, yalnizca esigin ustundekileri gelismis modele yollamak (iki asamali kaskad).
Soru: Kaskad, gelismis modele yakin basariyla daha dusuk gecikme saglar mi? Sonuc: kismen evet, bedelsiz degil.

## Veri
- IEEE-CIS Fraud Detection, yalnizca train_transaction.csv (Kaggle yarisma kosullariyla kullanildi, veri depoya eklenmedi): https://www.kaggle.com/competitions/ieee-fraud-detection/data
- 590.540 islem, 394 sutun, 20.663 fraud (%3,5). identity dosyasi kullanilmadi.
- Kronolojik bolme (bolmeler arasi bosluk YOK): train 413.378 (14.538 fraud), val 88.581 (3.042 fraud), test 88.581 (3.083 fraud).
- Kategorik kodlamalar yalnizca egitim bolmesinden ogrenildi. TransactionID, isFraud, TransactionDT ozellik degil; hour turetildi.
- Replay simulasyonu: veri gercek zamanli API degildir, gecmis islemler sirayla yeniden oynatilir.

## Yontem
- Hafif model (S1): HistGradientBoosting, 29 sutun, max_iter=60, max_depth=4. Egitim 3,1 sn.
- Gelismis model (S2): HistGradientBoosting, 392 sutun, max_iter=300, max_leaf_nodes=63. Egitim 163,6 sn.
- Yonlendirme: S1 skoru < T ise bypass, aksi halde S2. T=0,0166, yalnizca dogrulama setinde su kuralla secildi: recall@%2 kaybi < 0,01 olacak sekilde atlama oranini maksimize et. Test sonucuna bakilarak degistirilmedi.
- Karar kategorileri (prototip): skorun dogrulama dagilimindaki en ust %2'si incelemeye yonlendir, sonraki %3 ek dogrulama onerisi, kalani normal. Bypass edilenler normal.
- Baseline: B = yalniz hafif, C = yalniz gelismis. Kural tabanli baseline (A) yapilmadi.

## Sonuclar (test seti, tek calistirma)

| | PR-AUC | Recall@%2 |
|---|---|---|
| B hafif | 0,4418 | 0,350 |
| C gelismis | 0,5243 | 0,399 |
| Kaskad (T=0,0166) | 0,5158 | 0,399 |

Kaskad test islemlerinin %56,4'unu S2'ye gondermedi. Atlananlarin icinde tum fraudlarin %10,5'i (325/3083) var.

Farkli inceleme butcelerinde recall:

| Butce | Hafif | Gelismis | Kaskad | Kaskad-Gelismis |
|---|---|---|---|---|
| %1 | 0,235 | 0,251 | 0,251 | 0,000 |
| %2 | 0,350 | 0,399 | 0,399 | 0,000 |
| %5 | 0,525 | 0,566 | 0,557 | -0,008 |
| %10 | 0,623 | 0,691 | 0,680 | -0,011 |
| %20 | 0,725 | 0,820 | 0,805 | -0,015 |

Bootstrap (B=200, fark = kaskad - gelismis, %95 aralik): PR-AUC -0,0085 [-0,0116, -0,0060]; recall@%5 -0,0083 [-0,0125, -0,0039]; recall@%20 -0,0152 [-0,0227, -0,0083]. Araliklar sifiri icermiyor: kayip kucuk ama tutarli. Bu yuzden "ayni basari" denmiyor; dar inceleme butcesinde (%1-%2) olculen fark sifirdi.

## Gecikme: model cagrisi (surec ici, tek islem, OMP_NUM_THREADS=1, 3000 islem)

| | ort | p50 | p95 | p99 |
|---|---|---|---|---|
| B hafif | 1,61 ms | 1,58 | 1,76 | 1,98 |
| C gelismis | 4,26 ms | 4,22 | 4,49 | 4,76 |
| Kaskad | 3,22 ms | 1,64 | 5,98 | 6,20 |

Kaskad ortalama ve medyanda hizli, ama p95/p99 gelismis modelden kotu: S2'ye giden islem iki model suresini birden oder.

## Toplu cikarim (20.000 islem, 3 denemenin ortancasi, islem/sn)

| | B | C | Kaskad |
|---|---|---|---|
| Tek is parcacigi | 261.790 | 41.297 | 69.872 |
| Varsayilan is parcacigi | 603.772 | 104.062 | 152.268 |

## Uctan uca HTTP servisi (replay, 5000 islem, tek istemci, ardisik)

| | ort | p50 | p95 | p99 | islem/sn |
|---|---|---|---|---|---|
| Hafif | 2,57 ms | 2,41 | 3,17 | 6,39 | 339 |
| Gelismis | 5,15 ms | 4,98 | 6,07 | 8,48 | 180 |
| Kaskad | 4,01 ms | 2,42 | 6,95 | 8,67 | 227 |

Kaskad bu kosulda 5000 islemin %36,9'unu S2'ye gonderdi. Not: ilk HTTP surumunde yaklasik 50 ms sabit gecikme vardi; Nagle algoritmasi kapatilinca (TCP_NODELAY) uctan uca sure 49,9 ms den 3,9 ms ye dustu.

## Hata analizi (atlanan fraudlar)
Atlanan 325 fraudun yalnizca 11'i gelismis modelin en supheli %2'sine girerdi. Atlananlar ProductCD=W (268/325) ve debit kart (278/325) kumesinde yogun; W fraudlarinin %20,7'si, debit fraudlarinin %15,9'u atlandi. Iki degisken ic ice olabilir, hangisinin sebep oldugunu ayiramiyoruz. Sebep arastirilmadi.

## Sinirlamalar ve durustluk notlari
- Tum olcumler tek calistirma; gecikme icin guven araligi hesaplanmadi. Bootstrap yalnizca test orneklemini ve bagimsiz islem varsayimini kapsar, egitimi tekrarlamaz.
- Gelismis modelin PR-AUC'si dogrulamadan teste dustu (0,612 -> 0,524), hafif model sabit kaldi (0,434 -> 0,442). Nedeni arastirilmadi.
- Bolmeler arasinda bosluk yok; identity verisi kullanilmadi; V/C/D ozelliklerinin gelecek bilgisi sizdirip sizdirmadigi dogrulanamaz.
- Replay'de ozellikler hazir oldugu icin hafif modelin asil avantaji olabilecek ozellik cikarma maliyeti olculmedi.
- Kaskadin gecikme kazanci tek islemde sinirli (ortalama yaklasik %24, p95/p99 kotu); toplu islemde daha belirgin.
- LinuxONE baska bir mimariden daha hizlidir iddiasi YOKTUR; baska mimariyle karsilastirma yapilmadi. Tum sureler yalnizca bu VM icin gecerlidir.
- Kural tabanli baseline (A) yapilmadi; Docker, SHAP, identity join kapsam disi.

## LinuxONE kaniti
- uname -m: s390x. lscpu: Architecture s390x, Byte Order Big Endian, CPU(s) 3. RAM yaklasik 8 GB (/proc/meminfo MemTotal 8201236 kB).
- /etc/os-release: Red Hat Enterprise Linux 10.2. uname -a cekirdek bilgisi: Linux 6.8.0-147-generic s390x.
- Python 3.12.15, scikit-learn 1.9.0, numpy 2.0.2, pandas 3.0.6.
- Egitim, cikarim, API ve simulator ayni s390x VM'de calistirildi.
- VM'in ozel donanim hizlandiricilarini kullandigimiz iddia edilmemektedir.

## Nasil calistirilir
Veriyi (train_transaction.csv) Kaggle'dan indirip data/ altina koyun, sonra sirayla:
- python src/01_audit.py (denetim ve kronolojik bolme)
- python src/02_prep.py (ozellik sozlesmesi, X_full.npy)
- python src/03_train.py (iki model, dogrulama/test skorlari)
- python src/04_route.py (esik taramasi, yalnizca dogrulama)
- python src/05_test_eval.py (test basarisi, gercek sure)
- MODE=casc PORT=8000 python src/06_server.py (casc, adv veya light)
- python src/07_sim.py 5000 8000 (replay simulatoru)
- python src/08_errors.py, 09_budgets.py, 10_bootstrap.py (analizler)
Yollar /home/ibm-user/queueguard/ olarak sabit kodlanmistir. Modeller s390x (big-endian) uzerinde egitildi; baska mimaride joblib dosyalari acilmayabilir, o yuzden depoya eklenmedi.

## Zorunlu AI/ML laboratuvarlari
TODO: ekip tamamladiginda buraya yazilacak.

## Takim
Team AGEB.
