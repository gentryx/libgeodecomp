      SUBROUTINE KERNEL(SUM)
      IMPLICIT NONE
!      INTEGER, INTENT(OUT) :: SUM 
      INTEGER, INTENT(IN) :: SUM 

      INTEGER N
      COMMON /TEST/ N

!      REAL, DIMENSION(N), INTENT(IN) :: OLD_ALIVE
!      REAL, DIMENSION(N), INTENT(OUT) :: NEW_ALIVE
      
      INTEGER :: I
      
      DO I=1,N
         !do nothing
      END DO

!      SUM = N
      N=SUM

      END SUBROUTINE
