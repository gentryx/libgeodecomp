#ifndef LIBGEODECOMP_STORAGE_UPDATEFUNCTORMACROS_H
#define LIBGEODECOMP_STORAGE_UPDATEFUNCTORMACROS_H

// fixme: rename this to LIBGEODECOMP_CHUNK_THRESHOLD
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
            if (!concurrencySpec.preferFineGrainedParallelism()) {      \
                _Pragma("omp parallel for schedule(dynamic)")           \
                for (std::size_t c = 0; c < region.numPlanes(); ++c) {  \
                    typename Region<DIM>::StreakIterator e =            \
                        region.planeStreakIterator(c + 1);              \
                    typedef typename Region<DIM>::StreakIterator Iter;  \
                    for (Iter i = region.planeStreakIterator(c + 0);    \
                         i != e;                                        \
                         ++i) {                                         \
                        LGD_UPDATE_FUNCTOR_BODY;                        \
                    }                                                   \
                }                                                       \
    /**/
#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_3                         \
            } else {                                                    \
                typedef typename Region<DIM>::StreakIterator Iter;      \
                std::vector<Streak<DIM> > streaks;                      \
                streaks.resize(region.numStreaks());                    \
                int c = 0;                                              \
                for (Iter i = region.beginStreak(); i != region.endStreak(); ++i) { \
                    streaks[c] = *i;                                    \
                    ++c;                                                \
                }                                                       \
                _Pragma("omp parallel for schedule(static)")            \
                for (std::size_t j = 0; j < streaks.size(); ++j) {      \
                    Streak<DIM> *i = &streaks[j];                       \
                        LGD_UPDATE_FUNCTOR_BODY;                        \
                }                                                       \
            }                                                           \
    /**/
#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_4                         \
    /**/
#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_5                         \
    /**/
#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_6                         \
        }\
        return;\
        }
#else
#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_1
#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_2
#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_3
#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_4
#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_5
#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_6
#endif

#ifdef LIBGEODECOMP_WITH_HPX
// fixme: replace with executor parameter
#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_7                         \
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
#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_7
    /**/
#endif

#define LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_8                         \
    for (typename Region<DIM>::StreakIterator i = region.beginStreak(); \
         i != region.endStreak();                                       \
         ++i) {                                                         \
        LGD_UPDATE_FUNCTOR_BODY;                                        \
    }                                                                   \
    /**/


#endif
