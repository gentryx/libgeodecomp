      SUBROUTINE AVGDEPTH(N, DEPTH, AVG)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: N 
      REAL, DIMENSION(N) :: DEPTH
      REAL, INTENT(OUT) :: AVG
      
      INTEGER :: I

      AVG = 0.0
      DO I=1,N
         AVG = AVG + DEPTH(I)
      END DO
      AVG = AVG/N
      END SUBROUTINE
