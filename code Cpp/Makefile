# === Percorsi delle librerie statiche ===
TFLITE_INC = /home/jc/tflite-armv7-install/include
TFLITE_LIB = /home/jc/tflite-armv7-install/lib

#TF_INC = /home/jc/tflite_install/include

GEMM_INC = /home/jc/tflite-armv7-install/include

EIG_INC = /home/jc/tflite-armv7-install/include

FBS_INC = /home/jc/flatbuffers-armv7-install/include
FBS_LIB = /home/jc/flatbuffers-armv7-install/lib

ABSL_INC = /home/jc/abseil-install/include
ABSL_LIB = /home/jc/abseil-install/lib

OPENCV_INC = /home/jc/opencv-armv7-install/include/opencv4
OPENCV_LIB = /home/jc/opencv-armv7-install/lib

RETE_INC = /home/jc/TEST

#SYSROOT = /home/jc/rpi-sysroot


OBJS = fingerprint_stub.o cpuinfo_stub.o


# Cross-compiler
CXX = /home/jc/arm-gnu-toolchain-12.2.rel1-x86_64-arm-none-linux-gnueabihf/bin/arm-none-linux-gnueabihf-g++
#CXX = arm-linux-gnueabihf-g++
CXXFLAGS = -std=c++17 -O3 -I$(TFLITE_INC) -I$(FBS_INC) -I$(ABSL_INC) -I$(OPENCV_INC) -I$(EIG_INC) -I$(GEMM_INC) -I$(RETE_INC)

# Librerie TFLite + dipendenze statiche
TFLITE_OBJS = \
  $(TFLITE_LIB)/libtensorflow-lite.a \
  $(FBS_LIB)/libflatbuffers.a \
  $(TFLITE_LIB)/libruy_frontend.a \
  $(TFLITE_LIB)/libruy_trmul.a\
  $(TFLITE_LIB)/libruy_thread_pool.a\
  $(TFLITE_LIB)/libruy_context.a\
  $(TFLITE_LIB)/libruy_ctx.a\
  $(TFLITE_LIB)/libruy_context_get_ctx.a\
  $(TFLITE_LIB)/libruy_allocator.a\
  $(TFLITE_LIB)/libruy_apply_multiplier.a\
  $(TFLITE_LIB)/libruy_prepare_packed_matrices.a\
  $(TFLITE_LIB)/libruy_kernel_arm.a\
  $(TFLITE_LIB)/libruy_pack_arm.a\
  $(TFLITE_LIB)/libruy_profiler_profiler.a\
  $(TFLITE_LIB)/libruy_profiler_instrumentation.a\
  $(TFLITE_LIB)/libruy_system_aligned_alloc.a\
  $(TFLITE_LIB)/libruy_prepacked_cache.a\
  $(TFLITE_LIB)/libruy_wait.a\
  $(TFLITE_LIB)/libruy_tune.a\
  $(TFLITE_LIB)/libruy_block_map.a\
  $(TFLITE_LIB)/libruy_blocking_counter.a\
  $(TFLITE_LIB)/libruy_cpuinfo.a\
  $(TFLITE_LIB)/libruy_denormal.a\
  $(TFLITE_LIB)/libruy_have_built_path_for_avx.a\
  $(TFLITE_LIB)/libruy_have_built_path_for_avx2_fma.a\
  $(TFLITE_LIB)/libruy_have_built_path_for_avx512.a\
  $(TFLITE_LIB)/libruy_kernel_avx.a\
  $(TFLITE_LIB)/libruy_kernel_avx2_fma.a\
  $(TFLITE_LIB)/libruy_kernel_avx512.a\
  $(TFLITE_LIB)/libruy_pack_avx.a\
  $(TFLITE_LIB)/libruy_pack_avx2_fma.a \
  $(TFLITE_LIB)/libruy_pack_avx512.a \
  $(TFLITE_LIB)/libfft2d_fftsg2d.a  \
  $(TFLITE_LIB)/libfft2d_fftsg.a   \
  $(TFLITE_LIB)/libeight_bit_int_gemm.a \
  $(ABSL_LIB)/libabsl_time.a \
  $(ABSL_LIB)/libabsl_synchronization.a \
  $(ABSL_LIB)/libabsl_civil_time.a \
  $(ABSL_LIB)/libabsl_int128.a \
  $(ABSL_LIB)/libabsl_base.a \
  $(ABSL_LIB)/libabsl_strings.a \
  $(ABSL_LIB)/libabsl_throw_delegate.a \
  $(ABSL_LIB)/libabsl_flags_commandlineflag_internal.a \
  $(ABSL_LIB)/libabsl_log_internal_globals.a \
  $(ABSL_LIB)/libabsl_raw_logging_internal.a \
  $(ABSL_LIB)/libabsl_base.a\
  $(ABSL_LIB)/libabsl_flags_config.a \
  $(ABSL_LIB)/libabsl_log_internal_log_sink_set.a \
  $(ABSL_LIB)/libabsl_scoped_set_env.a \
  $(ABSL_LIB)/libabsl_city.a \
  $(ABSL_LIB)/libabsl_flags_internal.a \
  $(ABSL_LIB)/libabsl_log_internal_message.a \
  $(ABSL_LIB)/libabsl_spinlock_wait.a \
  $(ABSL_LIB)/libabsl_civil_time.a \
  $(ABSL_LIB)/libabsl_flags_marshalling.a \
  $(ABSL_LIB)/libabsl_log_internal_nullguard.a \
  $(ABSL_LIB)/libabsl_stacktrace.a \
  $(ABSL_LIB)/libabsl_cord.a \
  $(ABSL_LIB)/libabsl_flags_parse.a \
  $(ABSL_LIB)/libabsl_log_internal_proto.a \
  $(ABSL_LIB)/libabsl_status.a \
  $(ABSL_LIB)/libabsl_cord_internal.a \
  $(ABSL_LIB)/libabsl_flags_private_handle_accessor.a \
  $(ABSL_LIB)/libabsl_log_internal_structured_proto.a \
  $(ABSL_LIB)/libabsl_statusor.a \
  $(ABSL_LIB)/libabsl_cordz_functions.a \
  $(ABSL_LIB)/libabsl_flags_program_name.a \
  $(ABSL_LIB)/libabsl_log_severity.a \
  $(ABSL_LIB)/libabsl_strerror.a \
  $(ABSL_LIB)/libabsl_cordz_handle.a \
  $(ABSL_LIB)/libabsl_flags_reflection.a \
  $(ABSL_LIB)/libabsl_log_sink.a \
  $(ABSL_LIB)/libabsl_str_format_internal.a \
  $(ABSL_LIB)/libabsl_cordz_info.a \
  $(ABSL_LIB)/libabsl_flags_usage.a \
  $(ABSL_LIB)/libabsl_malloc_internal.a \
  $(ABSL_LIB)/libabsl_strings.a \
  $(ABSL_LIB)/libabsl_cordz_sample_token.a \
  $(ABSL_LIB)/libabsl_flags_usage_internal.a \
  $(ABSL_LIB)/libabsl_periodic_sampler.a \
  $(ABSL_LIB)/libabsl_strings_internal.a \
  $(ABSL_LIB)/libabsl_crc32c.a \
  $(ABSL_LIB)/libabsl_graphcycles_internal.a \
  $(ABSL_LIB)/libabsl_poison.a \
  $(ABSL_LIB)/libabsl_string_view.a \
  $(ABSL_LIB)/libabsl_crc_cord_state.a \
  $(ABSL_LIB)/libabsl_hash.a \
  $(ABSL_LIB)/libabsl_random_distributions.a \
  $(ABSL_LIB)/libabsl_symbolize.a \
  $(ABSL_LIB)/libabsl_crc_cpu_detect.a \
  $(ABSL_LIB)/libabsl_hashtablez_sampler.a \
  $(ABSL_LIB)/libabsl_random_internal_distribution_test_util.a \
  $(ABSL_LIB)/libabsl_synchronization.a \
  $(ABSL_LIB)/libabsl_crc_internal.a \
  $(ABSL_LIB)/libabsl_kernel_timeout_internal.a \
  $(ABSL_LIB)/libabsl_random_internal_platform.a \
  $(ABSL_LIB)/libabsl_time.a \
  $(ABSL_LIB)/libabsl_decode_rust_punycode.a \
  $(ABSL_LIB)/libabsl_leak_check.a \
  $(ABSL_LIB)/libabsl_random_internal_randen.a \
  $(ABSL_LIB)/libabsl_time_zone.a \
  $(ABSL_LIB)/libabsl_demangle_internal.a \
  $(ABSL_LIB)/libabsl_log_flags.a \
  $(ABSL_LIB)/libabsl_random_internal_randen_hwaes.a \
  $(ABSL_LIB)/libabsl_tracing_internal.a \
  $(ABSL_LIB)/libabsl_demangle_rust.a \
  $(ABSL_LIB)/libabsl_log_globals.a \
  $(ABSL_LIB)/libabsl_random_internal_randen_hwaes_impl.a \
  $(ABSL_LIB)/libabsl_utf8_for_code_point.a \
  $(ABSL_LIB)/libabsl_die_if_null.a \
  $(ABSL_LIB)/libabsl_log_initialize.a \
  $(ABSL_LIB)/libabsl_random_internal_randen_slow.a \
  $(ABSL_LIB)/libabsl_vlog_config_internal.a \
  $(ABSL_LIB)/libabsl_examine_stack.a \
  $(ABSL_LIB)/libabsl_log_internal_check_op.a \
  $(ABSL_LIB)/libabsl_random_internal_seed_material.a \
  $(ABSL_LIB)/libabsl_exponential_biased.a \
  $(ABSL_LIB)/libabsl_log_internal_conditions.a \
  $(ABSL_LIB)/libabsl_random_seed_gen_exception.a \
  $(ABSL_LIB)/libabsl_failure_signal_handler.a \
  $(ABSL_LIB)/libabsl_log_internal_fnmatch.a \
  $(ABSL_LIB)/libabsl_random_seed_sequences.a \
  $(ABSL_LIB)/libabsl_flags_commandlineflag.a \
  $(ABSL_LIB)/libabsl_log_internal_format.a \
  $(ABSL_LIB)/libabsl_raw_hash_set.a \
  $(ABSL_LIB)/libabsl_raw_logging_internal.a \
  $(ABSL_LIB)/libabsl_log_severity.a \
  $(ABSL_LIB)/libabsl_cord.a \
  $(ABSL_LIB)/libabsl_cord_internal.a \
  $(ABSL_LIB)/libabsl_stacktrace.a \
  $(ABSL_LIB)/libabsl_symbolize.a \
  $(ABSL_LIB)/libabsl_debugging_internal.a \
  $(ABSL_LIB)/libabsl_demangle_internal.a \
  $(ABSL_LIB)/libabsl_malloc_internal.a \
  $(ABSL_LIB)/libabsl_time_zone.a \
  $(ABSL_LIB)/libabsl_spinlock_wait.a \
  $(ABSL_LIB)/libabsl_graphcycles_internal.a \
  $(ABSL_LIB)/libabsl_int128.a \
  $(ABSL_LIB)/libabsl_str_format_internal.a \


# Librerie di sistema e OpenCV
SYS_LIBS = -lpthread -ldl -lm -latomic

OPENCV_STATIC_OBJS = \
  $(OPENCV_LIB)/libopencv_core.a \
  $(OPENCV_LIB)/libopencv_imgcodecs.a \
  $(OPENCV_LIB)/libopencv_imgproc.a \
  $(OPENCV_LIB)/libopencv_dnn.a \
  $(OPENCV_LIB)/libopencv_features2d.a \
  $(OPENCV_LIB)/libopencv_flann.a \
  $(OPENCV_LIB)/libopencv_gapi.a \
  $(OPENCV_LIB)/libopencv_calib3d.a \
  $(OPENCV_LIB)/libopencv_ml.a \
  $(OPENCV_LIB)/libopencv_photo.a \
  $(OPENCV_LIB)/libopencv_stitching.a \
  $(OPENCV_LIB)/libopencv_video.a \
  $(OPENCV_LIB)/libopencv_highgui.a\
  $(OPENCV_LIB)/libopencv_objdetect.a\
  $(OPENCV_LIB)/libopencv_videoio.a\
  $(OPENCV_LIB)/opencv4/3rdparty/libade.a \
  $(OPENCV_LIB)/opencv4/3rdparty/libittnotify.a \
  $(OPENCV_LIB)/opencv4/3rdparty/liblibopenjp2.a \
  $(OPENCV_LIB)/opencv4/3rdparty/liblibprotobuf.a \
  $(OPENCV_LIB)/opencv4/3rdparty/liblibjpeg-turbo.a \
  $(OPENCV_LIB)/opencv4/3rdparty/liblibpng.a \
  $(OPENCV_LIB)/opencv4/3rdparty/liblibtiff.a \
  $(OPENCV_LIB)/opencv4/3rdparty/liblibwebp.a \
  $(OPENCV_LIB)/opencv4/3rdparty/libtegra_hal.a \
  $(OPENCV_LIB)/opencv4/3rdparty/libzlib.a \



# File di input/output
#SRC = retewbcam.cpp
#SRC = test.cpp
#OUT = test
#OUT = retewbcam

SRC = main.cpp CameraHandler.cpp ClassifierInterpreter.cpp

OUT = classifier

# Regole
all: $(OUT)

$(OUT): $(SRC) $(OBJS)
	$(CXX) $(CXXFLAGS) -L$(TFLITE_LIB) -L$(FBS_LIB) -L$(ABSL_LIB) -L$(OPENCV_LIB) \
	$(SRC) $(OBJS) -o $@ $(TFLITE_OBJS) \
	-Wl,--start-group $(OPENCV_STATIC_OBJS) $(SYS_LIBS) -Wl,--end-group

clean:
	rm -f $(OUT)
