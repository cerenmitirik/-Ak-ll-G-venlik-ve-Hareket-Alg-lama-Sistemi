# Yapay Zekâ Destekli Akıllı Güvenlik ve Hareket Algılama Sistemi

## Proje Hakkında

Bu proje, ESP32-CAM üzerinden alınan canlı kamera görüntülerini yapay zekâ ve görüntü işleme algoritmalarıyla analiz eden bir güvenlik sistemidir.

Sistemde YOLOv8 nesne tespiti, OpenCV MOG2 arka plan çıkarma algoritması ve Kalman Filtresi birlikte kullanılarak insan, belirli hayvanlar ve hareketli nesneler tespit edilmektedir.

Algılanan durumlara göre MQTT protokolü üzerinden alarm bildirimleri gönderilmekte ve hedef canlı tespit edildiğinde otomatik video kaydı başlatılmaktadır.

## Kullanılan Teknolojiler

- **Python:** Ana uygulama ve görüntü işleme
- **ESP32-CAM:** Canlı kamera görüntüsü aktarımı
- **OpenCV:** Görüntü işleme ve hareket analizi
- **YOLOv8 Nano:** İnsan ve hayvan tespiti
- **MOG2:** Arka plan çıkarma ve hareketli bölgelerin belirlenmesi
- **Kalman Filtresi:** Nesne konumlarının tahmin edilmesi ve hareket takibi
- **NumPy:** Sayısal işlemler
- **MQTT (HiveMQ):** Alarm durumlarının iletilmesi
- **CUDA:** GPU üzerinde model çalıştırma

## Sistem Özellikleri

### 1. Yapay Zekâ ile Nesne Tespiti

YOLOv8 Nano modeli kullanılarak kamera görüntülerindeki belirli canlı sınıfları tespit edilir.

Kodda kullanılan sınıflar:
- İnsan (Person)
- Kedi (Cat)
- Köpek (Dog)

Tespit edilen nesneler görüntü üzerinde sınırlayıcı kutularla gösterilir. Belirlenen güven eşiğinin üzerindeki tespitler alarm oluşturur.

### 2. MOG2 ile Hareket Algılama

YOLOv8 tarafından hedef canlı tespit edilmediğinde MOG2 algoritması kullanılarak arka plan ile mevcut görüntü arasındaki değişimler incelenir.

- Hareketli bölgeler belirlenir.
- Gölge etkileri azaltılır.
- Morfolojik işlemlerle küçük görüntü gürültüleri temizlenir.
- Kontur analiziyle belirli alanın üzerindeki hareketli bölgeler seçilir.

### 3. Kalman Filtresi ile Hareket Takibi

Kalman Filtresi kullanılarak hareketli nesnelerin merkez koordinatları takip edilir.

Nesnenin önceki konumları ile güncel konumu karşılaştırılarak net yer değiştirme hesaplanır.

Bu yöntemle küçük kamera titreşimleri ve konum dalgalanmalarının yanlış alarm oluşturması azaltılmaya çalışılır.

### 4. Gece Modu

Kamera görüntüsünün ortalama parlaklık değeri hesaplanır.

Parlaklık belirlenen eşik değerinin altına düştüğünde sistem gece moduna geçerek hareket algılama işlemlerini pasif hale getirir.

### 5. MQTT ile Alarm Bildirimi

Sistem, alarm durumlarını HiveMQ MQTT broker üzerinden yayınlar.

- `1`: Alarm aktif
- `0`: Alarm pasif

Alarm durumu değiştiğinde MQTT üzerinden yeni durum gönderilir.

### 6. Otomatik ve Manuel Video Kaydı

Sistem iki farklı video kayıt yöntemini destekler.

**Otomatik kayıt:**
- YOLOv8 tarafından hedef canlı algılandığında kayıt başlatılır.
- Alarm durumu sona erdiğinde otomatik kayıt durdurulur.

**Manuel kayıt:**
- `V` tuşuyla video kaydı başlatılır veya durdurulur.
- `Q` tuşuyla uygulamadan çıkılır.

Video kayıtları tarih ve saat bilgisi içeren MP4 dosyaları olarak kaydedilir.

### 7. Performans Optimizasyonu

Gerçek zamanlı görüntü işlemede gecikmeyi azaltmak amacıyla kare atlama yöntemi kullanılmıştır.

YOLOv8 ve MOG2 analizleri her üç karede bir gerçekleştirilir. Ara karelerde son tespit sonuçları önbellekten kullanılır.

Ayrıca alarm durumunun sürekli değişmesini önlemek için ardışık sakin analiz sonuçlarını sayan bir debounce mekanizması uygulanmıştır.

## Sistem Çalışma Mantığı

1. ESP32-CAM üzerinden canlı video akışı alınır.
2. Görüntünün ortalama parlaklığı kontrol edilir.
3. YOLOv8 modeli ile hedef canlı sınıfları aranır.
4. Hedef canlı bulunmadığında MOG2 ile hareket analizi yapılır.
5. Kalman Filtresi ile hareketli nesnenin konumu takip edilir.
6. Yer değiştirme miktarı belirlenen eşikle karşılaştırılır.
7. Alarm durumu MQTT üzerinden yayınlanır.
8. Hedef canlı tespit edildiğinde otomatik video kaydı başlatılır.
9. Algılama sonuçları, FPS ve alarm bilgileri ekranda gösterilir.

## Kurulum

### Gereksinimler

- Python 3
- ESP32-CAM
- OpenCV
- NumPy
- Ultralytics
- Paho MQTT
- PyTorch

Gerekli Python kütüphaneleri:

```bash
pip install opencv-python numpy ultralytics paho-mqtt
```

GPU ile çalıştırmak için CUDA uyumlu PyTorch kurulumu gereklidir.

### Yapılandırma

Python dosyasındaki kamera adresi, MQTT broker ve topic bilgileri kullanılan ortama göre düzenlenmelidir.

```python
STREAM_URL = "http://KAMERA_IP:81/stream"
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "proje/ornek/alarm"
```

ESP32-CAM cihazının ağ üzerinden erişilebilir olması gerekir.

### Çalıştırma

```bash
python main.py
```

## Kullanılan Temel Parametreler

| Parametre | Değer |
|---|---|
| YOLO modeli | yolov8n.pt |
| YOLO güven eşiği | 0.45 |
| Minimum hareket alanı | 1000 piksel² |
| Yer değiştirme eşiği | 45 piksel |
| Işık eşiği | 45 |
| Kare atlama aralığı | 3 |
| Sakinlik eşiği | 10 analiz |
| MQTT portu | 1883 |

## Projenin Amacı

Bu projenin amacı, düşük maliyetli bir kamera sistemi ile yapay zekâ tabanlı nesne tespitini ve geleneksel görüntü işleme yöntemlerini birleştirerek gerçek zamanlı güvenlik takibi gerçekleştirmektir.

YOLOv8, MOG2 ve Kalman Filtresi yöntemlerinin birlikte kullanılmasıyla canlı tespiti, hareket analizi ve yanlış alarm azaltma işlemlerinin tek bir sistemde uygulanması hedeflenmiştir.
