using Microsoft.Win32;
using SkiaSharp;
using SkiaSharp.Views.Desktop;
using System.IO;
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

namespace POwB02
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private SKBitmap? bitmap = null;
        private SKBitmap? grayBitmap = null;
        private byte[,]? lbpImage = null;
        private int[]? lbpHistogram = null;

        public MainWindow()
        {
            InitializeComponent();
        }

        private void LoadImage_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new OpenFileDialog();
            dialog.Filter = "Obrazy (*.png;*.jpg;*.jpeg;*.bmp)|*.png;*.jpg;*.jpeg;*.bmp";

            if (dialog.ShowDialog() != true) return;

            bitmap = SKBitmap.Decode(dialog.FileName);
            if (bitmap == null)
            {
                MessageBox.Show("Nie udało się wczytać obrazu!");
                return;
            }

            InfoTextBlock.Text = $"Wczytano obraz: {bitmap.Width}x{bitmap.Height}";

            byte[,] gray = ConvertToGrayscale(bitmap);
            lbpImage = null;
            lbpHistogram = ComputeLBPHistogram(gray, out lbpImage);

            int max = lbpHistogram.Max();
            InfoTextBlock.Text += $"\nHistogram LBP policzony (max: {max})"; // maksymalna ilosc wystapienia danej wartosci

            OriginalCanvas.InvalidateVisual();
            GrayscaleCanvas.InvalidateVisual();
            LBPCanvas.InvalidateVisual();
            HistogramCanvas.InvalidateVisual();
        }

        private byte[,] ConvertToGrayscale(SKBitmap bmp)
        {
            int width = bmp.Width;
            int height = bmp.Height;
            byte[,] gray = new byte[width, height];
            grayBitmap = new SKBitmap(width, height);

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    var color = bmp.GetPixel(x, y);
                    byte g = (byte)((color.Red + color.Green + color.Blue) / 3);

                    gray[x, y] = g;
                    grayBitmap.SetPixel(x, y, new SKColor(g, g, g)); // ustaw pixel grayscale
                }
            }

            return gray;
        }

        private int[] ComputeLBPHistogram(byte[,] gray, out byte[,] lbpImage)
        {
            int width = gray.GetLength(0);
            int height = gray.GetLength(1);

            lbpImage = new byte[width, height]; // kopia do zapisu wartości LBP
            int[] histogram = new int[256];     // 8-bit LBP → 256 wartości

            int[,] offsets = {
        { -1, 0 },   // left
        { -1, 1 },   // bottom-left
        { 0, 1 },    // bottom
        { 1, 1 },    // bottom-right
        { 1, 0 },    // right
        { 1, -1 },   // top-right
        { 0, -1 },   // top
        { -1, -1 }   // top-left
    };

            int[] weights = { 128, 64, 32, 16, 8, 4, 2, 1 }; // odpowiadające potęgi 2

            for (int y = 1; y < height - 1; y++)
            {
                for (int x = 1; x < width - 1; x++)
                {
                    byte t = gray[x, y]; // centralny piksel
                    int code = 0;

                    for (int i = 0; i < 8; i++)
                    {
                        int nx = x + offsets[i, 0];
                        int ny = y + offsets[i, 1];

                        // jeśli sąsiad > t → 1, w przeciwnym wypadku 0
                        if (gray[nx, ny] > t)
                            code += weights[i];
                    }

                    lbpImage[x, y] = (byte)code;
                    histogram[code]++;
                }
            }

            return histogram;
        }

        private void OriginalCanvas_PaintSurface(object sender, SKPaintSurfaceEventArgs e)
        {
            var canvas = e.Surface.Canvas;
            canvas.Clear(SKColors.White);

            if (bitmap != null)
            {
                float scaleX = (float)e.Info.Width / bitmap.Width;
                float scaleY = (float)e.Info.Height / bitmap.Height;
                float scale = Math.Min(scaleX, scaleY);
                float offsetX = (e.Info.Width - bitmap.Width * scale) / 2f;
                float offsetY = (e.Info.Height - bitmap.Height * scale) / 2f;

                canvas.Save();
                canvas.Translate(offsetX, offsetY);
                canvas.Scale(scale);
                canvas.DrawBitmap(bitmap, 0, 0);
                canvas.Restore();
            }
        }

        private void GrayscaleCanvas_PaintSurface(object sender, SKPaintSurfaceEventArgs e)
        {
            var canvas = e.Surface.Canvas;
            canvas.Clear(SKColors.White);

            if (grayBitmap != null)
            {
                float scaleX = (float)e.Info.Width / grayBitmap.Width;
                float scaleY = (float)e.Info.Height / grayBitmap.Height;
                float scale = Math.Min(scaleX, scaleY);

                float offsetX = (e.Info.Width - grayBitmap.Width * scale) / 2f;
                float offsetY = (e.Info.Height - grayBitmap.Height * scale) / 2f;

                canvas.Save();
                canvas.Translate(offsetX, offsetY);
                canvas.Scale(scale);
                canvas.DrawBitmap(grayBitmap, 0, 0);
                canvas.Restore();
            }
        }

        private void LBPCanvas_PaintSurface(object sender, SKPaintSurfaceEventArgs e)
        {
            var canvas = e.Surface.Canvas;
            canvas.Clear(SKColors.White);

            if (lbpImage != null)
            {
                int width = lbpImage.GetLength(0);
                int height = lbpImage.GetLength(1);
                float scaleX = (float)e.Info.Width / width;
                float scaleY = (float)e.Info.Height / height;
                float scale = Math.Min(scaleX, scaleY);
                float offsetX = (e.Info.Width - width * scale) / 2f;
                float offsetY = (e.Info.Height - height * scale) / 2f;

                canvas.Save();
                canvas.Translate(offsetX, offsetY);
                canvas.Scale(scale);

                using var paint = new SKPaint();
                for (int y = 0; y < height; y++)
                    for (int x = 0; x < width; x++)
                    {
                        byte v = lbpImage[x, y];
                        paint.Color = new SKColor(v, v, v);
                        canvas.DrawPoint(x, y, paint);
                    }

                canvas.Restore();
            }
        }

        private void HistogramCanvas_PaintSurface(object sender, SKPaintSurfaceEventArgs e)
        {
            var canvas = e.Surface.Canvas;
            canvas.Clear(SKColors.White);

            if (lbpHistogram == null) return;

            int histLength = lbpHistogram.Length;
            int maxVal = lbpHistogram.Max();
            if (maxVal == 0) maxVal = 1;

            float barWidth = (float)e.Info.Width / histLength;

            using var paint = new SKPaint
            {
                Style = SKPaintStyle.Fill,
                Color = SKColors.SteelBlue
            };

            for (int i = 0; i < histLength; i++)
            {
                float barHeight = (float)lbpHistogram[i] / maxVal * e.Info.Height;
                canvas.DrawRect(i * barWidth, e.Info.Height - barHeight, barWidth, barHeight, paint);
            }
        }

        //Zadanie 2

        private void ProcessDatabase_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new CommonOpenFileDialog();
            dialog.IsFolderPicker = true;

            if (dialog.ShowDialog() != CommonFileDialogResult.Ok)
                return;

            string root = dialog.FileName;

            foreach (var personDir in Directory.GetDirectories(root))
            {
                string personName = System.IO.Path.GetFileName(personDir);
                string outputDir = System.IO.Path.Combine(personDir, "Histograms");
                Directory.CreateDirectory(outputDir);

                foreach (var file in Directory.GetFiles(personDir))
                {
                    if (!file.EndsWith(".jpg") && !file.EndsWith(".png") && !file.EndsWith(".bmp"))
                        continue;

                    var bmp = SKBitmap.Decode(file);
                    var gray = ConvertToGrayscale(bmp);
                    var hist = ComputeLBPHistogram(gray, out var lbpImg);

                    SaveHistogramAsPng(
                        hist,
                        System.IO.Path.Combine(
                            outputDir,
                            System.IO.Path.GetFileNameWithoutExtension(file) + "_hist.png"
                        )
                    );
                }
            }

            MessageBox.Show("Przetwarzanie zakończone!");
        }

        private void SaveHistogramAsPng(int[] histogram, string path)
        {
            int width = 512;     // 2px na każdą wartość (256*2)
            int height = 200;

            using var bmp = new SKBitmap(width, height);
            using var canvas = new SKCanvas(bmp);
            canvas.Clear(SKColors.White);

            int maxVal = histogram.Max();
            if (maxVal == 0) maxVal = 1;

            using var paint = new SKPaint
            {
                Color = SKColors.Black,
                Style = SKPaintStyle.Fill
            };

            for (int i = 0; i < 256; i++)
            {
                float x = i * 2;
                float h = (float)histogram[i] / maxVal * height;
                canvas.DrawRect(x, height - h, 2, h, paint);
            }

            using var img = SKImage.FromBitmap(bmp);
            using var data = img.Encode(SKEncodedImageFormat.Png, 100);
            File.WriteAllBytes(path, data.ToArray());
        }
    }
}