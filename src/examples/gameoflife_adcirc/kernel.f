      SUBROUTINE KERNEL(
     & N, 
     & ALIVE,
     & NUMNEIGHBORS,
     & NEIGHBORS
     & )
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: N 
      INTEGER, DIMENSION(N) :: ALIVE
      INTEGER, DIMENSION(N), INTENT(IN) :: NUMNEIGHBORS
      INTEGER, DIMENSION(20,N) :: NEIGHBORS

      INTEGER, DIMENSION(N) :: NEW_ALIVE            
      
      !LOOP VARIABLES
      INTEGER :: I
      INTEGER :: J

      !TEMP VARIABLES
      INTEGER :: SUM

      !LOOP OVER NODES
      DO I=1,N
         !PRINT*, "FORTRAN: NODE=", I, "NUMNEIGHBORS = ", NUMNEIGHBORS(I)
         NEW_ALIVE(I)=0
         IF(ALIVE(I).eq.1) THEN
            NEW_ALIVE(I)=0
         ELSE
            SUM=0
            DO J=1,NUMNEIGHBORS(I)
               !PRINT*, NEIGHBORS(J,I)
               SUM=SUM+ALIVE(NEIGHBORS(J,I))
            ENDDO
            IF(SUM.gt.0)THEN
               !PRINT*, "SUM=", SUM
               NEW_ALIVE(I)=1
            ENDIF
         ENDIF
!         PRINT*, "FORTRAN: I=",I," NEW_ALIVE=",NEW_ALIVE(I)
      ENDDO


      DO I=1,N
!         PRINT*, "FORTRAN: I=",I," NEW_ALIVE=",NEW_ALIVE(I)
         ALIVE(I)=NEW_ALIVE(I)
!         PRINT*, "FORTRAN: I=",I," ALIVE=",ALIVE(I)
      ENDDO

      END SUBROUTINE KERNEL
