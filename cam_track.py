import os
import time
from datetime import datetime
import cv2
import numpy as np
import paho.mqtt.client as mqtt
from ultralytics import YOLO

# ==========================================
# 1. AYARLAR VE YAPILANDIRMA
# ==========================================
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;5000000"

STREAM_URL = "http://10.42.0.18:81/stream"
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "proje/ceren/alarm"

ISIK_ESIGI = 45.0
MIN_HAREKET_ALANI = 1000
YER_DEGISTIRME_ESIGI = 45.0

# Asimetrik Debounce — güven skoru anlık sürekli düştüğü için  alarm susyor sonra tekrar. Dalgalanma yaratır. 
SAKINLIK_ESIGI = 10

# --- YENİ: Kare Atlama Ayarı ---
YZ_ATLAMA_ARALIGI = 3   # YOLO+MOG2 her 3 karede 1 çalışsın, arada önbellek kullanılsın. donma ve gecikme için kullan.
# tampon şişmesini kaldır, aradaki nesne kutuları önbellekte çizilecek

# kamera 15 kare verdi. 3 karede bir yolo çalıştı 15/3=5 kez analiz yapıldı. sakinlik eşiği 10 olsa 10/5 =2 saniye sürer.cisim kadrajdan çıksa bile 

HEDEF_SINIFLAR = [0, 15, 16]

# ==========================================
# 2. KALMAN TAKİPÇİSİ. nesnenin merkez noktasını all 8 kare geçtikten sonra net_yer_degistirme hesapla. eşiği geçiyors aharektli geçmiyorsa titreşim 
# ==========================================
class KalmanCentroidTracker:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2) 
        # 4 Durum Değişken, 2 Ölçüm Değişken. konumlar xy vx,vy hızlar
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    # ölçüm matris: 
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    # süreç gürültü q: sistemdeki fiziksel belirsizlik içinde modelin hıza makul düzeyde güvenmesini sağlar.
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
       # ölçüm gürültüsü r : anlık piksel oynamalarında süpheyle yakla tahmini dikkate al 
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 0.1
    # hata p : şu an tahmine ne kadar güveniyorum. zamnala predict ve correct çarıldıkca kendini günceller
        self.gecmis_noktalar = []
        self.kayip_kare_sayisi = 0
        self.baslatildi = False

    def guncelle(self, olcum_x, olcum_y)  :
        # konturun merkez koordinatlar al, , filtreden geçirip daha güvenilir konum ver, geçmiş noktalard tut
        self.kayip_kare_sayisi = 0
        olcum = np.array([[np.float32(olcum_x)], [np.float32(olcum_y)]], dtype=np.float32)
        if not self.baslatildi:
            self.kalman.statePre = np.array([[np.float32(olcum_x)], [np.float32(olcum_y)], [0.0], [0.0]], dtype=np.float32)
            self.kalman.statePost = np.array([[np.float32(olcum_x)], [np.float32(olcum_y)], [0.0], [0.0]], dtype=np.float32)
            self.baslatildi = True
            tahmin = (int(olcum_x), int(olcum_y))
        else:
            self.kalman.predict() # tahmin: bir önceki hız ve konuma bakra kcismin nereye gideceği 
            duzeltilmis = self.kalman.correct(olcum) # yeni ölçümü alır .kendi tahmini ve kameranın ölçümünü taratarak en doğru konumu belirle
            tahmin = (int(duzeltilmis[0][0]), int(duzeltilmis[1][0]))

        self.gecmis_noktalar.append(tahmin)
        if len(self.gecmis_noktalar) > 20:
            self.gecmis_noktalar.pop(0)
        return tahmin

    def nesne_kayboldu(self):
        self.kayip_kare_sayisi += 1
        if self.kayip_kare_sayisi > 5:
            self.gecmis_noktalar.clear()
            self.baslatildi = False

    def net_yer_degistirme(self):
        if len(self.gecmis_noktalar) < 8:
            return 0.0
        ilk_x, ilk_y = self.gecmis_noktalar[0]
        son_x, son_y = self.gecmis_noktalar[-1]
        return float(np.sqrt((son_x - ilk_x)**2 + (son_y - ilk_y)**2))

# ==========================================
# 3. BAŞLATMA
# ==========================================
print("[i] YOLOv8 Modeli GPU'ya yukleniyor...")
yolo_model = YOLO("yolov8n.pt").to('cuda')

#print("[i] YOLOv8 Modeli CPU'ya yukleniyor...")
#yolo_model = YOLO("yolov8n.pt").to('cpu')

bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=350, varThreshold=40, detectShadows=True)
# piksel seviyesinde gölge silme. "burada değişen ne var". Gerçek nesneleri beyaz (255), gölgeleri ise gri (127). yani gölge nesne değil ,ışık engelleyen 
# history: arka planı öğrenirken hafızada tut ,varthreshold: pikselin farklı sayılması için ne kadar sapması gerektiği. sapanlar yeni nesne
tracker = KalmanCentroidTracker()

try:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
except AttributeError:
    client = mqtt.Client()

try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
    print(f"[+] MQTT Broker baglantisi kuruldu: {MQTT_BROKER}")
except Exception as e:
    print(f"[-] MQTT Hatasi: {e}")
    exit()

cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("[-] Hata: ESP32-CAM video akisi acilamadi!")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if frame_width == 0 or frame_height == 0:
    frame_width, frame_height = 320, 240

print("[+] Guvenlik Sistemi Calisiyor.")
print("[i] 'v' tusu: Video Kaydini Baslat / Durdur")
print("[i] 'q' tusu: Cikis")

last_state = None
aktif_durum = '0'
sakin_kare_sayaci = SAKINLIK_ESIGI

# --- YENİ: Kare sayacı ve önbellek değişkenleri - yolo çalışmadığı halde  kararları güncelelme son hesaplananın ekranda tut.
kare_sayaci = 0
son_alarm_nedeni = "Sistem Sakin"
son_anlik_tehdit = False
son_kutu_bilgisi = []        # YOLO kutularını önbellekte tutmak için: [(x1,y1,x2,y2,isim,guven), ...]
son_merkez = None
son_dr = 0.0

# FPS ölçümü için
fps_zamani = time.time()
fps_sayac = 0
anlik_fps = 0

video_writer = None
kayit_yapiliyor = False
otomatik_kayit_mi = False

def video_kaydini_baslat(otomatik=False):
    global video_writer, kayit_yapiliyor, otomatik_kayit_mi
    if not kayit_yapiliyor:
        zaman_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        on_ek = "kayit_oto_insan" if otomatik else "kayit_manuel"
        dosya_adi = f"{on_ek}_{zaman_str}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Kamera FPS değerine uygun olarak 15-20 FPS verilir
        video_writer = cv2.VideoWriter(dosya_adi, fourcc, 15.0, (frame_width, frame_height))
        kayit_yapiliyor = True
        otomatik_kayit_mi = otomatik
        
        tur_etiketi = "[AUTO-REC]" if otomatik else "[MANUAL-REC]"
        print(f"\n{tur_etiketi} Video kaydi BASLADI: {dosya_adi}")

def video_kaydini_durdur():
    global video_writer, kayit_yapiliyor, otomatik_kayit_mi
    if kayit_yapiliyor and video_writer is not None:
        video_writer.release()
        video_writer = None
        kayit_yapiliyor = False
        tur_etiketi = "[AUTO-STOP]" if otomatik_kayit_mi else "[MANUAL-STOP]"
        otomatik_kayit_mi = False
        print(f"{tur_etiketi} Video diske kaydedildi ve KAPATILDI.\n")

# ==========================================
# 4. ANA DÖNGÜ
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame = cv2.flip(frame, 1)
    kare_sayaci += 1

    # Bu karede YZ (YOLO+MOG2) çalışacak mı?
    yz_calissin_mi = (kare_sayaci % YZ_ATLAMA_ARALIGI == 0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ortalama_parlaklik = np.mean(gray)

    # ----------------------------------------------------
    # ADIM 1: GECE MODU — HER KAREDE kontrol edilir (ucuz, atlanmaz)
    # ----------------------------------------------------
    if ortalama_parlaklik < ISIK_ESIGI:
        son_anlik_tehdit = False
        aktif_durum = '0'
        sakin_kare_sayaci = SAKINLIK_ESIGI
        tracker.nesne_kayboldu()
        son_alarm_nedeni = f"GECE MODU: Pasif ({ortalama_parlaklik:.1f})"
        son_kutu_bilgisi = []
        cv2.putText(frame, son_alarm_nedeni, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    elif yz_calissin_mi: 
        # ----------------------------------------------------
        # ADIM 2: sakin_kare_Sayaci yalnızca yolo ya da mog  çalıştığında algılanır. 

# Kare 1 (ekrana basılır): YZ ÇALIŞMAZ → önbellek gösterilir, sayaç DOKUNULMAZ
#Kare 2 (ekrana basılır): YZ ÇALIŞMAZ → önbellek gösterilir, sayaç DOKUNULMAZ
#Kare 3 (ekrana basılır): YZ ÇALIŞIR   → tehdit yok → sakin_kare_sayaci += 1  (1)

#6 kare geçse sayaç tehit yok sayaç 2 ye çıkar noş kareleri saymaz. her ekran karesinde değil. kaç kere ardışık olarak yz tehdit görmedi ? diye bakıyoruz. ekran karesi saysaydık hızla artardı debounce anlaösız  olur 
        anlik_tehdit = False
        hedef_canli_tespit = False
        alarm_nedeni = "Sistem Sakin"
        en_buyuk_cnt = None
        dr = 0.0
        yeni_kutu_bilgisi = []

        yolo_results = yolo_model(frame, imgsz=320, stream=True, verbose=False, device=0)
        for r in yolo_results:
            for box in r.boxes:
                sinif_id = int(box.cls[0])
                guven = float(box.conf[0])
                if sinif_id in HEDEF_SINIFLAR and guven > 0.45:
                    hedef_canli_tespit = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # kutunun köşe noktalarını verir. array olarak gelir float gelir int yap.
                    yeni_kutu_bilgisi.append((x1, y1, x2, y2, yolo_model.names[sinif_id], guven))

        if hedef_canli_tespit: # yolo hedef bulamadığında çalışıyo. 
            anlik_tehdit = True
            alarm_nedeni = "ALARM: Hedef Canli Algilandi!"
            tracker.nesne_kayboldu()
        else:
            fg_mask = bg_subtractor.apply(frame)  # foregroudn mask  :ön plan maskesi. , bg_substractor: arka planı karşılaştıran o anki farkı hesaplar.
            # o anki görüntüyü al, gri tonlamalı yap , 0= arka plan 255 = ön plan/hareket (gerçek nesne), 127 = gölge
            _, thresh = cv2.threshold(fg_mask, 167, 255, cv2.THRESH_BINARY)
            # gölgeleri siler. gerçek nesne olan beyazları bırak. 127 den büyük olanlar 255 yap küçük olanları 0 yap. 127 den büyükler gerçek nesne
# Önce ufak parazitleri temizle (Open)

# Sonra cismin içindeki boşlukları doldur ve gövdeyi birleştir (Close)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            #önce ereazyon küçültme sonra dilaiton büyütme ile . beyaz gürültüleri sl. büyük gerçek nesneleri etkileme 
            konturlar, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # beyaz bölgelerin konturlarını bul.RETR_EXTERNAL = sadece en dıştaki , CHAIN_APPROX_SIMPLE adece uç noktalarını tut. 
            max_alan = 0

            for cnt in konturlar:
                alan = cv2.contourArea(cnt) 
                # kontrurun kapladığı piksel alan.. alan maxtan küçükse yok say. 
                if alan > MIN_HAREKET_ALANI and alan > max_alan:
                    max_alan = alan
                    en_buyuk_cnt = cnt

            if en_buyuk_cnt is not None:
                M = cv2.moments(en_buyuk_cnt) 
                if M["m00"] != 0: # konturun toplam alanı
                    cx = int(M["m10"] / M["m00"]) # m10 ve m01 = x ve y koordinatlarının ağırlıklı toplamları
                    cy = int(M["m01"] / M["m00"])
                    merkez = tracker.guncelle(cx, cy)
                    dr = tracker.net_yer_degistirme()
                    son_merkez = merkez
                    son_dr = dr

                    if dr > YER_DEGISTIRME_ESIGI:
                        anlik_tehdit = True
                        alarm_nedeni = f"ALARM: Cisim Hareketi! (Dr:{dr:.1f}px)"
                    else:
                        alarm_nedeni = f"Titresim/Sabit Golge Elendi (Dr:{dr:.1f}px)"
            else:
                tracker.nesne_kayboldu()

        # --- Sonuçları önbelleğe kaydet (aradaki karelerde kullanılacak) ---
        son_anlik_tehdit = anlik_tehdit
        son_alarm_nedeni = alarm_nedeni
        son_kutu_bilgisi = yeni_kutu_bilgisi

        if anlik_tehdit:
            sakin_kare_sayaci = 0
            aktif_durum = '1'
        else:
            sakin_kare_sayaci += 1
            # tehdit kesildiğinde kod hemen 0 göndermez. sayacı he rsakin karede +1 artırır. ortam gerçekten sakin olduğunda 10 a ukaştığında 0 gönderir. algılama kayıpları yutulmuş oldu.
            if sakin_kare_sayaci >= SAKINLIK_ESIGI:
                aktif_durum = '0'
    # kaç gerçek karar sorusuna cevap vermesi gerekiyor .kaç ekran karesi geçtiye değil 

    else:
        # ----------------------------------------------------
        # ADIM 3: ARA KARE — hiçbir AI çalışmaz, önbellek gösterilir
        # ----------------------------------------------------
        pass  # aktif_durum, son_alarm_nedeni, son_kutu_bilgisi olduğu gibi kalır

    # ----------------------------------------------------
    # ADIM 4: ÇİZİM — her karede önbellekteki kutular çizilir
    # ----------------------------------------------------
    for (x1, y1, x2, y2, isim, guven) in son_kutu_bilgisi:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"{isim} {guven:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if son_merkez is not None:
        renk = (0, 0, 255) if son_dr > YER_DEGISTIRME_ESIGI else (255, 255, 0)
        cv2.circle(frame, son_merkez, 6, renk, -1)

    # ----------------------------------------------------
    # ADIM 5: MQTT DURUM BİLDİRİMİ — her karede kontrol edilir
    # ----------------------------------------------------
    if aktif_durum != last_state:
        client.publish(MQTT_TOPIC, aktif_durum)
        print(f"[>] HiveMQ Bildirimi: {aktif_durum} -> {son_alarm_nedeni}")
        last_state = aktif_durum
    if son_anlik_tehdit and "Hedef Canli" in son_alarm_nedeni and not kayit_yapiliyor:
        video_kaydini_baslat(otomatik=True)
    elif aktif_durum == '0' and kayit_yapiliyor and otomatik_kayit_mi:
        video_kaydini_durdur()
    fps_sayac += 1
    if time.time() - fps_zamani >= 1.0:
        anlik_fps = fps_sayac
        fps_sayac = 0
        fps_zamani = time.time()
    durum_renk = (0, 0, 255) if aktif_durum == '1' else (0, 255, 0)
    cv2.putText(frame, son_alarm_nedeni, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, durum_renk, 2)
    cv2.putText(frame, f"FPS: {anlik_fps}", (20, frame_height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if kayit_yapiliyor:
        # Manuel kayıt kırmızı, otomatik kayıt turuncu daire gösterir
        rec_renk = (0, 140, 255) if otomatik_kayit_mi else (0, 0, 255)
        rec_etiket = "REC [OTO]" if otomatik_kayit_mi else "REC [MAN]"
        
        cv2.circle(frame, (frame_width - 25, 25), 7, rec_renk, -1)
        cv2.putText(frame, rec_etiket, (frame_width - 110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rec_renk, 2)
        video_writer.write(frame)

    cv2.imshow("Akilli Guvenlik Takip Sistemi", frame)

    if 'fg_mask' in locals() and fg_mask is not None:
       cv2.imshow("1. Ham MOG2 Maskesi (fg_mask)", fg_mask)
    if 'thresh' in locals() and thresh is not None:
       cv2.imshow("2. Morfolojik Temizlenmis (thresh)", thresh)

    tus = cv2.waitKey(1) & 0xFF
    if tus == ord('v'):
        if not kayit_yapiliyor:
            video_kaydini_baslat()
        else:
            video_kaydini_durdur()
    elif tus == ord('q'):
        client.publish(MQTT_TOPIC, '0')
        break

if kayit_yapiliyor:
    video_kaydini_durdur()

cap.release()
client.loop_stop()
client.disconnect()
cv2.destroyAllWindows()
print("[i] Guvenlik sistemi guvenle kapatildi.")
