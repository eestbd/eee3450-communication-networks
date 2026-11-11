통신네트워크 HW1 소스 코드 및 실행 방법
2022142233 이우현

이 묶음은 소스 코드·실행 방법·실험 근거를 포함한다.
3페이지 최종 보고서 PDF는 별도로 제출한다.

1. 실행 환경 및 설치
Python 3.12.10, numpy 2.4.6, matplotlib 3.11.2에서 실행하였다.
시뮬레이션은 numpy, 그림 생성은 matplotlib을 사용한다.
ZIP을 풀고 이 README와 aimd_sim.py가 있는 폴더에서 실행한다.

python -m pip install -r requirements.txt

2. 보고서의 주 비교 재현
아래 명령은 100,000 RTT 중 초기 20,000 RTT를 제외하고 seed 0~9를 사용한다.
A는 결정론적이므로 1회만 실행한다. A/B/C는 같은 네트워크를 사용한다.

python -X utf8 run_experiments.py --output reproduced_main --steps 100000 --warmup 20000 --modes A B C --main-only
python -X utf8 plot_results.py --results reproduced_main
python -X utf8 audit_results.py --results reproduced_main

A = 일반 손실 기반 AIMD
B = 초기 고정 알림 확률안(진단용)
C = 보고서 (3)의 최종 제안: 실제 block throughput에 따른 가변 알림 확률
보고서의 주 비교는 A와 C이다.

출력의 summary.csv에서 scenario=main, mode=A/C를 확인한다.
일반 throughput: (22.500, 67.499, 22.500) packets/RTT
제안 throughput: (19.008, 70.180, 25.757) packets/RTT
제안 PF 오차 평균: (0.097, 1.128, 0.860)%
제안 link 활용률: R1=99.10%, R2=99.48%

3. 보고서의 추가 장기 실험 재현
제안 C만 300,000 RTT 실행하고 초기 100,000 RTT를 제외한다.

python -X utf8 check_stability.py --output reproduced_stability
python -X utf8 plot_stability.py --results reproduced_stability

이 추가 구간의 throughput은 (19.023, 70.729, 25.899) packets/RTT이다.
주 비교의 100k 측정 구간과 구분한다.

4. 파일 및 원자료
aimd_sim.py: 공통 네트워크, queue, scheduler, A/B/C 제어, 측정 함수
run_experiments.py: 검증, 확률 AIMD 이론 확인, 반복 실험 및 CSV 출력
check_stability.py: C의 추가 장기 실험
audit_results.py: 저장된 trace의 capacity/AIMD/OR 규칙과 실제 적응 경로 재현 검사
plot_results.py, plot_stability.py: 저장된 CSV의 그림 생성
reference_results/main_100k/: 보고서 주 비교의 seed별 결과·집계·검증 메타데이터
reference_results/extended_300k/: 별도 C 장기 실험의 결과·block 평균
figures/: 보고서 그림 1·2의 원본 PNG. plot_results.py는 같은 원자료를 방식별 그림으로 작성한다.

trace_*.csv는 실행 시 생성되며 슬롯별 cwnd, 송신·수신량, flight, queue,
실제 라우터 출력, 실제 drop량, 제어 mark, AIMD 반응, 알림 확률을 포함한다.
summary.csv의 _mean은 seed 평균, _seed_sd는 seed 간 표본 표준편차다.
PF 오차는 seed별 절대 상대오차를 구한 뒤 평균한 값이다.
U는 seed별 동일 측정 구간의 평균 throughput으로 구한 Σln(x_i)다.

5. 모델과 설정
기본 RTT=1인 실수 단위 fluid model이며, R1 다음 R2를 service한다.
R1 각 독립 입력 queue는 180, R2 각 queue는 90 packets다.
같은 비가중 max-min work-conserving scheduler를 사용한다.
송신 시도량은 max(cwnd-flight,0), ACK/drop 통지는 다음 슬롯에 적용한다.
제어 mark와 실제 overflow 손실은 OR로 합쳐 한 번만 반감한다.
C의 block 길이는 500 RTT, 확률 갱신 이득은 0.5, p 범위는 [1e-5,0.5]이다.
미확인량·queue 데이터를 유지하며 실제 queue overflow 외의 drop은 없다.
큐 대기는 추가 지연을 만든다. TCP 재전송·RTO·순서 보장은 생략하였다.

6. 실행 시 주의
기존 실험 폴더를 덮어쓰지 않도록 --output 이름은 새 이름을 사용한다.
audit_results.py도 기존 independent_audit.json이 있으면 중단한다.
Python -O 옵션은 assert 검증을 끄므로 사용하지 않는다.
용량 90/45를 변경하면 PF 해와 가격도 다시 계산해야 한다.
