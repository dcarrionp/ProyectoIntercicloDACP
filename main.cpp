#include <iostream>
#include <filesystem>
#include <iomanip>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkExtractImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Uso: " << argv[0] << " volumen.nii.gz mascara.nii.gz carpeta_salida" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string volumenPath = argv[1];
    const std::string mascaraPath = argv[2];
    const std::string carpetaSalida = argv[3];
    fs::create_directories(carpetaSalida);

    constexpr unsigned int Dimension = 3;
    using PixelType = float;
    using Image3DType = itk::Image<PixelType, Dimension>;
    using Image2DType = itk::Image<PixelType, 2>;
    using ReaderType = itk::ImageFileReader<Image3DType>;
    using ExtractFilterType = itk::ExtractImageFilter<Image3DType, Image2DType>;
    using RescaleFilterType = itk::RescaleIntensityImageFilter<Image2DType, itk::Image<unsigned char, 2>>;

    // Lectura del volumen y la máscara
    ReaderType::Pointer volReader = ReaderType::New();
    volReader->SetFileName(volumenPath);
    volReader->Update();
    Image3DType::Pointer volumen = volReader->GetOutput();

    ReaderType::Pointer maskReader = ReaderType::New();
    maskReader->SetFileName(mascaraPath);
    maskReader->Update();
    Image3DType::Pointer mascara = maskReader->GetOutput();

    Image3DType::SizeType size = volumen->GetLargestPossibleRegion().GetSize();
    std::cout << "Procesando " << size[2] << " slices..." << std::endl;

    // Video
    cv::VideoWriter video(carpetaSalida + "/video_slices.avi", cv::VideoWriter::fourcc('M','J','P','G'), 5, cv::Size(size[0], size[1]));

    // Archivo CSV para estadísticas
    std::ofstream statsFile(carpetaSalida + "/estadisticas.csv");
    statsFile << "Slice,Area,Media,Minimo,Maximo\n";

    for (int z = 0; z < static_cast<int>(size[2]); ++z) {
        Image3DType::IndexType start = {0, 0, z};
        Image3DType::SizeType sliceSize = {size[0], size[1], 0};
        Image3DType::RegionType sliceRegion;
        sliceRegion.SetSize(sliceSize);
        sliceRegion.SetIndex(start);

        auto extractAndRescaleSlice = [&](Image3DType::Pointer input) -> cv::Mat {
            ExtractFilterType::Pointer extractor = ExtractFilterType::New();
            extractor->SetExtractionRegion(sliceRegion);
            extractor->SetInput(input);
            extractor->SetDirectionCollapseToSubmatrix();
            extractor->Update();

            RescaleFilterType::Pointer rescaler = RescaleFilterType::New();
            rescaler->SetInput(extractor->GetOutput());
            rescaler->SetOutputMinimum(0);
            rescaler->SetOutputMaximum(255);
            rescaler->Update();

            const auto* buffer = rescaler->GetOutput()->GetBufferPointer();
            if (!buffer) return {};

            auto region = rescaler->GetOutput()->GetLargestPossibleRegion();
            auto width = region.GetSize()[0];
            auto height = region.GetSize()[1];

            return cv::Mat(height, width, CV_8UC1, const_cast<void*>(static_cast<const void*>(buffer))).clone();
        };

        cv::Mat volMat = extractAndRescaleSlice(volumen);
        cv::Mat maskMat = extractAndRescaleSlice(mascara);

        if (volMat.empty() || maskMat.empty()) {
            std::cerr << "⚠️ Slice Z=" << z << " inválido o vacío. Se omite." << std::endl;
            continue;
        }

        // 🔹 Preprocesamiento: filtros
        cv::GaussianBlur(volMat, volMat, cv::Size(5, 5), 1.0);
        cv::medianBlur(maskMat, maskMat, 3);

        // 🔹 Morfología: apertura y cierre
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(maskMat, maskMat, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(maskMat, maskMat, cv::MORPH_CLOSE, kernel);

        // 🔹 Estadísticas
        int area = cv::countNonZero(maskMat);
        cv::Scalar meanVal;
        double minVal, maxVal;
        cv::meanStdDev(volMat, meanVal, cv::noArray(), maskMat);
        cv::minMaxLoc(volMat, &minVal, &maxVal, nullptr, nullptr, maskMat);
        statsFile << z << "," << area << "," << meanVal[0] << "," << minVal << "," << maxVal << "\n";

        // 🔹 Colorización
        cv::Mat colorVol;
        cv::cvtColor(volMat, colorVol, cv::COLOR_GRAY2BGR);
        for (int y = 0; y < maskMat.rows; ++y) {
            for (int x = 0; x < maskMat.cols; ++x) {
                if (maskMat.at<uchar>(y, x) > 10) {
                    colorVol.at<cv::Vec3b>(y, x) = {0, 0, 255};  // rojo
                }
            }
        }

        // 🔹 Guardar imagen y agregar al video
        std::ostringstream filename;
        filename << carpetaSalida << "/slice_" << std::setw(3) << std::setfill('0') << z << ".png";
        cv::imwrite(filename.str(), colorVol);
        video.write(colorVol);
    }

    statsFile.close();
    video.release();
    std::cout << "✅ Procesamiento completado. Imágenes, estadísticas y video guardados en: " << carpetaSalida << std::endl;
    return EXIT_SUCCESS;
}