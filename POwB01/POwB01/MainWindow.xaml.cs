using System.Linq;
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
using Microsoft.Win32;
using SkiaSharp;
using SkiaSharp.Views.Desktop;
using Svg.Skia;


namespace POwB01
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        SKPicture? svgPicture = null;
        SKBitmap? bitmap = null;

        private SKPoint? selectionStart = null;
        private SKRect selectionRect = SKRect.Empty;

        public MainWindow()
        {
            InitializeComponent();
        }

        private void LoadImage_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new OpenFileDialog();
            dialog.Filter = "Obrazy (*.png;*.jpg;*.jpeg;*.bmp;*.svg)|*.png;*.jpg;*.jpeg;*.bmp;*.svg";

            if (dialog.ShowDialog() != true)
                return;

            string extension = System.IO.Path.GetExtension(dialog.FileName).ToLower();
            MessageBox.Show($"Wybrano plik: {dialog.FileName}\nRozszerzenie: {extension}");

            if (extension == ".svg")
            {
                var svg = new SKSvg();
                svg.Load(dialog.FileName);
                svgPicture = svg.Picture;
                bitmap = null;

                if (svgPicture == null)
                    MessageBox.Show("SVG nie został wczytany!");

            }
            else
            {
                bitmap = SKBitmap.Decode(dialog.FileName);
                svgPicture = null;

                if (bitmap == null)
                    MessageBox.Show("Bitmapa nie została wczytana!");
            }


            // przerysowanie canvasa
            CanvasSkia.InvalidateVisual();
        }


        private void CanvasSkia_PaintSurface(object sender, SKPaintSurfaceEventArgs e)
        {
            var canvas = e.Surface.Canvas;

            canvas.Clear(SKColors.White);


            if (svgPicture != null)
            {
                float scaleX = e.Info.Width / svgPicture.CullRect.Width;
                float scaleY = e.Info.Height / svgPicture.CullRect.Height;
                float scale = Math.Min(scaleX, scaleY);

                float offsetX = (e.Info.Width - svgPicture.CullRect.Width * scale) / 2f;
                float offsetY = (e.Info.Height - svgPicture.CullRect.Height * scale) / 2f;

                canvas.Save();
                canvas.Translate(offsetX, offsetY);
                canvas.Scale(scale);
                canvas.DrawPicture(svgPicture);
                canvas.Restore();
            }
            else if (bitmap != null)
            {
                float scaleX = (float)e.Info.Width / bitmap.Width;
                float scaleY = (float)e.Info.Height / bitmap.Height;
                float scale = Math.Min(scaleX, scaleY);

                float offsetX = (e.Info.Width - bitmap.Width * scale) / 2f;
                float offsetY = (e.Info.Height - bitmap.Height * scale) / 2f;

                var destRect = new SKRect(offsetX, offsetY, offsetX + bitmap.Width * scale, offsetY + bitmap.Height * scale);
                canvas.DrawBitmap(bitmap, destRect);
            }

            if (!selectionRect.IsEmpty)
            {
                using var paint = new SKPaint
                {
                    Color = SKColors.Red.WithAlpha(128),
                    Style = SKPaintStyle.Stroke,
                    StrokeWidth = 2
                };
                canvas.DrawRect(selectionRect, paint);
            }
        }

        private void CanvasSkia_MouseDown(object sender, MouseEventArgs e)
        {
            if(bitmap == null) { return;
            }
            var pos = e.GetPosition(this);
            selectionStart = new SKPoint((float)pos.X, (float)pos.Y);
        }

        private void CanvasSkia_MouseMove(object sender, MouseEventArgs e)
        {
            if(bitmap == null || selectionStart == null || e.LeftButton != MouseButtonState.Pressed) 
            { return;}

            var pos = e.GetPosition(CanvasSkia);
            var end = new SKPoint((float)pos.X, (float)pos.Y);

            selectionRect = new SKRect(
        Math.Min(selectionStart.Value.X, end.X),
        Math.Min(selectionStart.Value.Y, end.Y),
        Math.Max(selectionStart.Value.X, end.X),
        Math.Max(selectionStart.Value.Y, end.Y)
    );

            CanvasSkia.InvalidateVisual();


        }

        private void CanvasSkia_MouseUp(object sender, MouseButtonEventArgs e)
        {
            if (bitmap == null || selectionStart == null) return;

            ComputeStats(selectionRect); // <- tu liczymy statystyki
            selectionStart = null;
            CanvasSkia.InvalidateVisual();
        }

        private void ComputeStats(SKRect rect)
        {
            if(bitmap == null)
            {
                MessageBox.Show("bitmap empty");
                return;
            }

            if (selectionRect.IsEmpty) 
            {
                MessageBox.Show("selectRect empty");
                return; 
            }

            int xStart = (int)(rect.Left * bitmap.Width / (float)CanvasSkia.ActualWidth);
            int yStart = (int)(rect.Top * bitmap.Height / (float)CanvasSkia.ActualHeight);
            int xEnd = (int)(rect.Right * bitmap.Width / (float)CanvasSkia.ActualWidth);
            int yEnd = (int)(rect.Bottom * bitmap.Height / (float)CanvasSkia.ActualHeight);

            xStart = Math.Clamp(xStart, 0, bitmap.Width - 1);
            yStart = Math.Clamp(yStart, 0, bitmap.Height - 1);
            xEnd = Math.Clamp(xEnd, 0, bitmap.Width - 1);
            yEnd = Math.Clamp(yEnd, 0, bitmap.Height - 1);

            var rValues = new List<byte>();
            var gValues = new List<byte>();
            var bValues = new List<byte>();


            // dodanie pikseli do list
            for (int y = yStart; y <= yEnd; y++)
            {
                for (int x = xStart; x <= xEnd; x++)
                {
                    var color = bitmap.GetPixel(x, y);
                    rValues.Add(color.Red);
                    gValues.Add(color.Green);
                    bValues.Add(color.Blue);
                }
            }

            if (rValues.Count == 0 || gValues.Count == 0 || bValues.Count == 0)
            {
                MessageBox.Show("Wybrany obszar nie zawiera pikseli!");
                return;
            }

            // obliczenie średniej, mediany, wariancji i odchylenia standardowego
            double Mean(List<byte> vals) => vals.Average(v => (double)v);
            double Median(List<byte> vals) => vals.OrderBy(v => v).ElementAt(vals.Count / 2);
            double Variance(List<byte> vals, double mean) => vals.Select(v => Math.Pow(v - mean, 2)).Average();
            double StdDev(double variance) => Math.Sqrt(variance);

            double rMean = Mean(rValues);
            double gMean = Mean(gValues);
            double bMean = Mean(bValues);

            double rMedian = Median(rValues);
            double gMedian = Median(gValues);
            double bMedian = Median(bValues);

            double rVariance = Variance(rValues, rMean);
            double gVariance = Variance(gValues, gMean);
            double bVariance = Variance(bValues, bMean);

            double rStd = StdDev(rVariance);
            double gStd = StdDev(gVariance);
            double bStd = StdDev(bVariance);

            // na razie możemy wyświetlić w konsoli, potem w TextBlockach
            SredniaTextBlock.Text = $"Srednia:\n R: {rMean:F1}, G: {gMean:F1}, B: {bMean:F1}";
            MedianaTextBlock.Text = $"Mediana: \n R: {rMedian:F1}, G: {gMedian:F1}, B: {bMedian:F1}";
            WariancjaTextBlock.Text = $"Wariancja: \n R: {rVariance:F1}, G: {gVariance:F1}, B: {bVariance:F1}";
            OdchylenieTextBlock.Text = $"Odchylenie Standardowe: \n R: {rStd:F1}, G: {gStd:F1}, B: {bStd:F1}";
        }
    }
}