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

void aplicarTecnicasPorSeparado(const cv::Mat& vol, const cv::Mat& mask, int z, const std::string& path) {
    char buf[32];
    sprintf(buf, "%03d", z);

    cv::Mat thresholded;
    cv::threshold(vol, thresholded, 100, 255, cv::THRESH_BINARY);
    cv::imwrite(path + "/resultados_threshold/slice_" + buf + "_threshold.png", thresholded);

    cv::Mat stretched;
    cv::normalize(vol, stretched, 0, 255, cv::NORM_MINMAX);
    cv::imwrite(path + "/resultados_stretch/slice_" + buf + "_stretch.png", stretched);

    cv::Mat inRangeMask;
    cv::inRange(vol, 120, 200, inRangeMask);
    cv::imwrite(path + "/resultados_inrange/slice_" + buf + "_inrange.png", inRangeMask);

    cv::Mat logic_or, logic_xor, logic_not;
    cv::bitwise_or(vol, mask, logic_or);
    cv::bitwise_xor(vol, mask, logic_xor);
    cv::bitwise_not(mask, logic_not);
    cv::imwrite(path + "/resultados_or/slice_" + buf + "_or.png", logic_or);
    cv::imwrite(path + "/resultados_xor/slice_" + buf + "_xor.png", logic_xor);
    cv::imwrite(path + "/resultados_not/slice_" + buf + "_not.png", logic_not);

    cv::Mat edges;
    cv::Canny(vol, edges, 100, 200);
    cv::imwrite(path + "/resultados_canny/slice_" + buf + "_canny.png", edges);
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Uso: " << argv[0] << " volumen.nii.gz mascara.nii.gz carpeta_salida" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string volumenPath = argv[1];
    const std::string mascaraPath = argv[2];
    const std::string carpetaSalida = argv[3];
    fs::create_directories(carpetaSalida);

    fs::create_directories(carpetaSalida + "/resultados_full");
    fs::create_directories(carpetaSalida + "/resultados_masked");
    fs::create_directories(carpetaSalida + "/resultados_threshold");
    fs::create_directories(carpetaSalida + "/resultados_stretch");
    fs::create_directories(carpetaSalida + "/resultados_inrange");
    fs::create_directories(carpetaSalida + "/resultados_or");
    fs::create_directories(carpetaSalida + "/resultados_xor");
    fs::create_directories(carpetaSalida + "/resultados_not");
    fs::create_directories(carpetaSalida + "/resultados_canny");

    constexpr unsigned int Dimension = 3;
    using PixelType = float;
    using Image3DType = itk::Image<PixelType, Dimension>;
    using Image2DType = itk::Image<PixelType, 2>;
    using ReaderType = itk::ImageFileReader<Image3DType>;
    using ExtractFilterType = itk::ExtractImageFilter<Image3DType, Image2DType>;
    using RescaleFilterType = itk::RescaleIntensityImageFilter<Image2DType, itk::Image<unsigned char, 2>>;

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

    cv::VideoWriter video(carpetaSalida + "/video_slices.avi", cv::VideoWriter::fourcc('M','J','P','G'), 5, cv::Size(size[0], size[1]));
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

        if (volMat.empty() || maskMat.empty()) continue;

        cv::GaussianBlur(volMat, volMat, cv::Size(5, 5), 1.0);
        cv::medianBlur(maskMat, maskMat, 3);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(maskMat, maskMat, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(maskMat, maskMat, cv::MORPH_CLOSE, kernel);

        int area = cv::countNonZero(maskMat);
        cv::Scalar meanVal;
        double minVal, maxVal;
        cv::meanStdDev(volMat, meanVal, cv::noArray(), maskMat);
        cv::minMaxLoc(volMat, &minVal, &maxVal, nullptr, nullptr, maskMat);
        statsFile << z << "," << area << "," << meanVal[0] << "," << minVal << "," << maxVal << "\n";

        // Aplicar técnicas en todos los slices
        aplicarTecnicasPorSeparado(volMat, maskMat, z, carpetaSalida);

        cv::Mat colorFull;
        cv::cvtColor(volMat, colorFull, cv::COLOR_GRAY2BGR);
        for (int y = 0; y < maskMat.rows; ++y)
            for (int x = 0; x < maskMat.cols; ++x)
                if (maskMat.at<uchar>(y, x) > 10)
                    colorFull.at<cv::Vec3b>(y, x) = {0, 0, 255};

        std::ostringstream nameFull;
        nameFull << carpetaSalida << "/resultados_full/slice_" << std::setw(3) << std::setfill('0') << z << "_full.png";
        cv::imwrite(nameFull.str(), colorFull);

        cv::Mat volMasked;
        cv::bitwise_and(volMat, maskMat, volMasked);
        cv::Mat colorMasked;
        cv::cvtColor(volMasked, colorMasked, cv::COLOR_GRAY2BGR);
        for (int y = 0; y < maskMat.rows; ++y)
            for (int x = 0; x < maskMat.cols; ++x)
                if (maskMat.at<uchar>(y, x) > 10)
                    colorMasked.at<cv::Vec3b>(y, x) = {0, 0, 255};

        std::ostringstream nameMasked;
        nameMasked << carpetaSalida << "/resultados_masked/slice_" << std::setw(3) << std::setfill('0') << z << "_masked.png";
        cv::imwrite(nameMasked.str(), colorMasked);

        video.write(colorMasked);
    }

    statsFile.close();
    video.release();
    std::cout << "✅ Procesamiento completado y resultados organizados en subcarpetas: " << carpetaSalida << std::endl;
    return EXIT_SUCCESS;
}
