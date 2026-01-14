using Microsoft.Win32;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using IOPath = System.IO.Path;

namespace POwB3
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private BitmapSource _originalBitmap;

        public MainWindow()
        {
            InitializeComponent();
        }

        // wczytanie obrazu
        private void BtnLoad_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog
            {
                Filter = "Obrazy (*.jpg;*.jpeg;*.png)|*.jpg;*.jpeg;*.png",
                Title = "Wybierz zdjęcie do segmentacji"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                // wczytanie i konwersja na format Bgra32
                BitmapImage bitmap = new BitmapImage(new Uri(openFileDialog.FileName));
                _originalBitmap = new FormatConvertedBitmap(bitmap, PixelFormats.Bgra32, null, 0);

                ImgOriginal.Source = _originalBitmap;
                ImgResult.Source = null; 
                TxtStatus.Text = $"Wczytano: {IOPath.GetFileName(openFileDialog.FileName)}";
            }
        }

        // przycisk segmentacji
        private async void BtnStart_Click(object sender, RoutedEventArgs e)
        {
            if (_originalBitmap == null)
            {
                MessageBox.Show("nie wczytano obrazu", "FATAL MISTAKE", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            // parametry
            int k = (int)SliderK.Value; //ilosc grup
            if (!int.TryParse(TxtIterations.Text, out int maxIterations)) maxIterations = 10; // iteracje

            BtnStart.IsEnabled = false;
            ProgressB.Visibility = Visibility.Visible;
            TxtStatus.Text = "doin the job";

            // dane z oryginalu
            int width = _originalBitmap.PixelWidth;
            int height = _originalBitmap.PixelHeight;
            int stride = width * 4;
            byte[] pixels = new byte[height * stride];
            _originalBitmap.CopyPixels(pixels, stride, 0);

            // async
            try
            {
                byte[] resultPixels = await Task.Run(() =>
                    PerformKMeans(pixels, k, maxIterations));

                // tworzenie wyniku
                ImgResult.Source = BitmapSource.Create(
                    width, height, 96, 96, PixelFormats.Bgra32, null, resultPixels, stride);

                TxtStatus.Text = "Segmentacja zakończona sukcesem.";
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Wystąpił błąd: {ex.Message}");
            }
            finally
            {
                BtnStart.IsEnabled = true;
                ProgressB.Visibility = Visibility.Collapsed;
            }
        }

        // k-means
        private byte[] PerformKMeans(byte[] pixelData, int k, int maxIterations)
        {
            int numPixels = pixelData.Length / 4;
            // rgb bajty, ilosc pixeli razy bajt
            (byte R, byte G, byte B)[] pixels = new (byte R, byte G, byte B)[numPixels];

            for (int i = 0; i < numPixels; i++)
            {
                pixels[i] = (pixelData[i * 4 + 2], pixelData[i * 4 + 1], pixelData[i * 4]); // formatowanie danych
            }

            // tworzenie centroidow
            Random rand = new Random();
            var centroids = new (double R, double G, double B)[k]; //sumy odleglosci dla centroidow
            for (int i = 0; i < k; i++)
                centroids[i] = pixels[rand.Next(numPixels)]; //losowanie kolorow z obrazu na poczatek centroidow

            int[] assignments = new int[numPixels];// numer klastra dla piksela

            // iteracje
            for (int iter = 0; iter < maxIterations; iter++)
            {
                bool changed = false;

                // przypisanie pikseli do klastrow
                for (int i = 0; i < numPixels; i++)
                {
                    int bestK = 0;
                    double minDistance = double.MaxValue;

                    for (int j = 0; j < k; j++)
                    {
                        // liczenie odleglosci od srodka

                        // Kwadrat odległości euklidesowej (szybszy niż Math.Sqrt)
                        double dist = Math.Pow(pixels[i].R - centroids[j].R, 2) +
                                      Math.Pow(pixels[i].G - centroids[j].G, 2) +
                                      Math.Pow(pixels[i].B - centroids[j].B, 2);

                        if (dist < minDistance)
                        {
                            minDistance = dist;
                            bestK = j;
                        }
                    }

                    if (assignments[i] != bestK) //nadpisanie grupy
                    {
                        assignments[i] = bestK;
                        changed = true;
                    }
                }

                if (!changed) break;

                // update srodkow klastrów/ srednich kolorow
                var sumR = new double[k]; var sumG = new double[k]; var sumB = new double[k];
                var counts = new int[k];

                for (int i = 0; i < numPixels; i++)// dodawanie wartosci pikseli
                {
                    int c = assignments[i];
                    sumR[c] += pixels[i].R;
                    sumG[c] += pixels[i].G;
                    sumB[c] += pixels[i].B;
                    counts[c]++;
                }

                for (int j = 0; j < k; j++)
                {
                    if (counts[j] > 0)
                    {
                        //liczenie srednich srodkow centroidow na podstawie sum przez ilosc
                        centroids[j] = (sumR[j] / counts[j], sumG[j] / counts[j], sumB[j] / counts[j]);
                    }
                }
            }

            //koncowa tablica bajtow
            byte[] resultData = new byte[pixelData.Length];
            for (int i = 0; i < numPixels; i++)
            {
                int c = assignments[i];
                resultData[i * 4] = (byte)centroids[c].B;
                resultData[i * 4 + 1] = (byte)centroids[c].G;
                resultData[i * 4 + 2] = (byte)centroids[c].R;
                resultData[i * 4 + 3] = 255; // alpha
            }

            return resultData;
        }
    }
}