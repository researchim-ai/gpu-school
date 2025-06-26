cmake_minimum_required(VERSION 3.20)

project(gemm_clblast LANGUAGES C CXX)

find_package(OpenCL REQUIRED)
find_package(CLBlast QUIET)

if(CLBlast_FOUND)
    add_executable(gemm_clblast src/gemm_clblast.cpp)
    if(TARGET CLBlast::clblast)
        target_link_libraries(gemm_clblast PRIVATE OpenCL::OpenCL CLBlast::clblast)
    elseif(TARGET CLBlast::CLBlast)
        target_link_libraries(gemm_clblast PRIVATE OpenCL::OpenCL CLBlast::CLBlast)
    elseif(TARGET clblast)
        target_link_libraries(gemm_clblast PRIVATE OpenCL::OpenCL clblast)
    else()
        message(FATAL_ERROR "Не удалось определить имя импортируемой цели CLBlast. Проверьте установку.")
    endif()
    target_compile_features(gemm_clblast PUBLIC cxx_std_17)

    if(PROJECT_IS_TOP_LEVEL)
        enable_testing()
    endif()
    add_test(NAME gemm_clblast COMMAND gemm_clblast 512)
else()
    message(WARNING "CLBlast не найден — модуль 07-libraries будет пропущен.\n" 
            "Установите пакет libclblast-dev или соберите из исходников: https://github.com/CNugteren/CLBlast")
endif() 