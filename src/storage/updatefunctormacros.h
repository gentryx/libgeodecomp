#ifndef LIBGEODECOMP_STORAGE_UPDATEFUNCTORMACROS_H
#define LIBGEODECOMP_STORAGE_UPDATEFUNCTORMACROS_H

#if !defined(LGD_CHUNK_THRESHOLD)
#define LGD_CHUNK_THRESHOLD 0
#endif

#ifdef LIBGEODECOMP_WITH_THREADS
#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_1                         \
    if (concurrencySpec.enableOpenMP() &&                               \
        !modelThreadingSpec.hasOpenMP()) {                              \
        if (concurrencySpec.preferStaticScheduling()) {                 \
            _Pragma("omp parallel for schedule(static)")                \
            for (std::size_t c = 0; c < region.numPlanes(); ++c) {      \
                typename Region<DIM>::StreakIterator e =                \
                    region.planeStreakIterator(c + 1);                  \
                typedef typename Region<DIM>::StreakIterator Iter;      \
                for (Iter i = region.planeStreakIterator(c + 0);        \
                     i != e;                                            \
                     ++i) {                                             \
                    LGD_UPDATE_FUNCTOR_BODY;                            \
                }                                                       \
            }                                                           \
    /**/
#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_2                         \
        } else {                                                        \
            _Pragma("omp parallel for schedule(dynamic)")               \
            for (std::size_t c = 0; c < region.numPlanes(); ++c) {      \
                typename Region<DIM>::StreakIterator e =                \
                    region.planeStreakIterator(c + 1);                  \
                typedef typename Region<DIM>::StreakIterator Iter;      \
                for (Iter i = region.planeStreakIterator(c + 0);        \
                     i != e;                                            \
                     ++i) {                                             \
                    LGD_UPDATE_FUNCTOR_BODY;                            \
                }                                                       \
            }                                                           \
        }                                                               \
        return;                                                         \
    }                                                                   \
    /**/
#else
#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_1
#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_2
#endif

#ifdef LIBGEODECOMP_WITH_HPX
// fixme: replace with executor parameter
#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_3                         \
    if (concurrencySpec.enableHPX() && !modelThreadingSpec.hasHPX()) {  \
        std::vector<hpx::future<void> > updateFutures;                  \
        updateFutures.reserve(region.numPlanes());                      \
        typedef typename Region<DIM>::StreakIterator Iter;              \
        Iter begin = region.beginStreak();                              \
        Iter end = region.endStreak();                                  \
        const int chunkThreshold = LGD_CHUNK_THRESHOLD;                 \
        while(begin != end) {                                           \
            Iter next = begin;                                          \
            int chunkLength = 0;                                        \
            while(next != end) {                                        \
                chunkLength += next->length();                          \
                ++next;                                                 \
                if(chunkLength >= chunkThreshold || next == end) {      \
                    updateFutures << hpx::async(                        \
                        [&](Iter i, Iter end) {                         \
                            for(; i != end; ++i) {                      \
                                LGD_UPDATE_FUNCTOR_BODY;                \
                            }                                           \
                        }, begin, next);                                \
                    break;                                              \
                }                                                       \
            }                                                           \
            begin = next;                                               \
        }                                                               \
        hpx::wait_all(updateFutures);                                   \
                                                                        \
        return;                                                         \
    }                                                                   \
    /**/
#else
#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_3
    /**/
#endif

#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_4                         \
    for (typename Region<DIM>::StreakIterator i = region.beginStreak(); \
         i != region.endStreak();                                       \
         ++i) {                                                         \
        LGD_UPDATE_FUNCTOR_BODY;                                        \
    }                                                                   \
    /**/


#endif
