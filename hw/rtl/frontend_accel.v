// Neuro-GPT front-end accelerator: the fused 22x25 spatio-temporal convolution, int8 x int8 -> int32.
//
// For each output time step t (0..475) and filter o (0..39):
//     acc[t][o] = sum_{c<22} sum_{k<25} W[o][c][k] * X[c][t+k]
// LANES = 40 parallel MAC lanes (one per filter) share one input sample per cycle, so a time step takes
// 22*25 = 550 cycles and a 2-s chunk takes 476*550 = 261,800 cycles. Requantization, ELU and pooling stay
// on the host CPU (they are 0.1% of the work). Weights are a ROM initialised from w.hex.
//
// Host interface: write the int8 chunk X (address c*500 + t) through x_we/x_addr/x_din, pulse start,
// wait for done, then read acc[t][o] at address t*40 + o through y_addr/y_dout (one-cycle read latency).
module frontend_accel #(
    parameter LANES = 40,
    parameter C = 22,
    parameter T = 500,
    parameter K = 25,
    parameter T1 = T - K + 1,
    parameter WFILE = "w.hex"
) (
    input  wire               clk,
    input  wire               rst,
    input  wire               start,
    input  wire               x_we,
    input  wire [13:0]        x_addr,
    input  wire signed [7:0]  x_din,
    input  wire [14:0]        y_addr,
    output reg  signed [31:0] y_dout,
    output reg                done
);
    localparam NI = C * K;  // 550 multiply-accumulates per lane per time step

    // ---------------- memories ----------------
    reg signed [7:0]         xmem [0:C*T-1];
    reg [LANES*8-1:0]        wmem [0:NI-1];
    reg signed [31:0]        ymem [0:T1*LANES-1];
    initial $readmemh(WFILE, wmem);

    always @(posedge clk) if (x_we) xmem[x_addr] <= x_din;
    always @(posedge clk) y_dout <= ymem[y_addr];

    // ---------------- stage 0: address generation ----------------
    reg        running;
    reg [8:0]  t;
    reg [4:0]  c, k;
    reg [13:0] x_raddr;
    reg [9:0]  w_raddr;
    reg        s0_valid, s0_first, s0_last;
    reg [8:0]  s0_t;

    always @(posedge clk) begin
        if (rst) begin
            running <= 1'b0; t <= 0; c <= 0; k <= 0; s0_valid <= 1'b0;
        end else begin
            s0_valid <= 1'b0;
            if (start && !running) begin
                running <= 1'b1; t <= 0; c <= 0; k <= 0;
            end else if (running) begin
                x_raddr  <= c * T + t + k;
                w_raddr  <= c * K + k;
                s0_valid <= 1'b1;
                s0_first <= (c == 0) && (k == 0);
                s0_last  <= (c == C - 1) && (k == K - 1);
                s0_t     <= t;
                if (k == K - 1) begin
                    k <= 0;
                    if (c == C - 1) begin
                        c <= 0;
                        if (t == T1 - 1) running <= 1'b0;
                        else t <= t + 1;
                    end else c <= c + 1;
                end else k <= k + 1;
            end
        end
    end

    // ---------------- stage 1: synchronous memory reads ----------------
    reg signed [7:0]  x_q;
    reg [LANES*8-1:0] w_q;
    reg               s1_valid, s1_first, s1_last;
    reg [8:0]         s1_t;
    always @(posedge clk) begin
        x_q      <= xmem[x_raddr];
        w_q      <= wmem[w_raddr];
        s1_valid <= rst ? 1'b0 : s0_valid;
        s1_first <= s0_first;
        s1_last  <= s0_last;
        s1_t     <= s0_t;
    end

    // ---------------- stage 2: 40 MAC lanes ----------------
    reg signed [31:0] acc  [0:LANES-1];
    reg signed [31:0] snap [0:LANES-1];
    reg               snap_valid;
    reg [8:0]         snap_t;
    genvar l;
    generate
        for (l = 0; l < LANES; l = l + 1) begin : lane
            wire signed [7:0]  w   = w_q[l*8 +: 8];
            wire signed [15:0] p   = w * x_q;
            wire signed [31:0] sum = (s1_first ? 32'sd0 : acc[l]) + p;
            always @(posedge clk) begin
                if (s1_valid) begin
                    acc[l] <= sum;
                    if (s1_last) snap[l] <= sum;
                end
            end
        end
    endgenerate

    // ---------------- write-back: drain 40 results while the next step computes ----------------
    reg [5:0]  wo;
    reg        draining;
    reg [8:0]  steps_done;
    always @(posedge clk) begin
        if (rst) begin
            snap_valid <= 1'b0; draining <= 1'b0; wo <= 0; steps_done <= 0; done <= 1'b0;
        end else begin
            if (start && !running) begin done <= 1'b0; steps_done <= 0; end
            snap_valid <= s1_valid && s1_last;
            if (s1_valid && s1_last) snap_t <= s1_t;
            if (snap_valid) begin draining <= 1'b1; wo <= 0; end
            else if (draining) begin
                ymem[snap_t * LANES + wo] <= snap[wo];
                if (wo == LANES - 1) begin
                    draining <= 1'b0;
                    steps_done <= steps_done + 1;
                    if (steps_done == T1 - 1) done <= 1'b1;
                end else wo <= wo + 1;
            end
        end
    end
endmodule
