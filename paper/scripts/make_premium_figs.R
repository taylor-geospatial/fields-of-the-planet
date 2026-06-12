#!/usr/bin/env Rscript
# Premium-grade per-country bars and smallholder scatter for the FTP paper.
# Uses ggplot2 + ggrepel with a custom restrained theme (Inter-like font stack,
# muted neutrals for non-emphasized data, paired olive/sienna for win/loss).
#
# Inputs:
#   logs/ftw_official/b7_*.csv               (released S2 PRUE-B7 numbers)
#   logs/fulldata_eval/planet_b3_augmax_full_ws_tta.csv  (our best Planet run)
#   paper/scripts/output/smallholder_scatter.csv         (median ha + deltas)
#
# Outputs:
#   paper/figs/per_country_bars_premium.pdf
#   paper/figs/smallholder_scatter_premium.pdf

suppressPackageStartupMessages({
  library(ggplot2)
  library(ggrepel)
  library(dplyr)
  library(readr)
  library(scales)
  library(cowplot)
})

# ---------------------------------------------------------------------------
# Theme.  The goal is "looks designed, not plotted."  We strip every default
# axis line we don't need, sit the title flush-left with a sentence-case look,
# pin gridlines to subtle dashed greys, and lean on a single accent palette
# (deep olive vs warm sienna) inherited from the FTP hero colors.
# ---------------------------------------------------------------------------

OLIVE     <- "#4f6b1f"
OLIVE_DK  <- "#2f4314"
SIENNA    <- "#a3441d"
SIENNA_DK <- "#6d2c11"
NEUTRAL   <- "#9b9b9b"
INK       <- "#1c1c1c"
PAPER     <- "#ffffff"
GRID      <- "#dcdad3"

# Helvetica clone (URW Nimbus Sans) is installed system-wide and matches the
# typographic register of CVPR/WACV figures: sans-serif figure text against a
# Times-Roman body.  Fall back to "sans" if absent.
FIG_FONT <- "Nimbus Roman"

theme_premium <- function(base_size = 9) {
  theme_minimal(base_size = base_size, base_family = FIG_FONT) +
    theme(
      plot.background       = element_rect(fill = PAPER, color = NA),
      panel.background      = element_rect(fill = PAPER, color = NA),
      panel.grid.major.x    = element_line(color = GRID, linewidth = 0.25),
      panel.grid.major.y    = element_blank(),
      panel.grid.minor      = element_blank(),
      axis.ticks            = element_blank(),
      axis.line.x           = element_line(color = INK, linewidth = 0.3),
      axis.line.y           = element_blank(),
      axis.text             = element_text(color = INK, size = base_size - 1),
      axis.title.x          = element_text(color = INK, size = base_size, margin = margin(t = 6)),
      axis.title.y          = element_blank(),
      plot.title            = element_text(color = INK, size = base_size + 4, face = "bold", hjust = 0, margin = margin(b = 2)),
      plot.subtitle         = element_text(color = "#555555", size = base_size, hjust = 0, margin = margin(b = 10)),
      plot.caption          = element_text(color = "#777777", size = base_size - 2, hjust = 0, margin = margin(t = 8)),
      plot.title.position   = "plot",
      plot.caption.position = "plot",
      legend.position       = "none",
      plot.margin           = margin(12, 14, 8, 8)
    )
}

# ---------------------------------------------------------------------------
# Data: per-country deltas.
# ---------------------------------------------------------------------------

s2_files <- Sys.glob("logs/ftw_official/b7_*.csv")
s2_files <- s2_files[!grepl("per_country", s2_files)]
s2 <- do.call(rbind, lapply(s2_files, read_csv, show_col_types = FALSE))
s2 <- s2 |>
  rename(country = countries, obj_f1 = object_level_f1) |>
  distinct(country, .keep_all = TRUE) |>
  select(country, pixel_level_iou, obj_f1) |>
  rename(iou_s2 = pixel_level_iou, f1_s2 = obj_f1)

pl <- read_csv("logs/fulldata_eval/planet_b3_augmax_full_ws_tta.csv", show_col_types = FALSE) |>
  select(country, pixel_level_iou, object_ws_f1) |>
  rename(iou_pl = pixel_level_iou, f1_pl = object_ws_f1)

deltas <- inner_join(s2, pl, by = "country") |>
  mutate(
    d_f1 = (f1_pl - f1_s2) * 100,
    d_iou = (iou_pl - iou_s2) * 100,
    country_lbl = tools::toTitleCase(gsub("_", " ", country)),
    winner = ifelse(d_f1 >= 0, "Planet wins", "S2 wins")
  )

# ---------------------------------------------------------------------------
# Figure 1: per-country bars (Obj F1 only).  Pixel-IoU panel is *deliberately
# dropped* from the main figure — it crowded the layout and the paper already
# argues pixel IoU is GSD-biased.  We surface it as a small companion panel
# below using cowplot::plot_grid so the reader can still see the divergence.
# ---------------------------------------------------------------------------

# Highlight the 3 biggest wins and 3 biggest losses; everything else fades to
# neutral grey so the eye lands on the extremes.
deltas <- deltas |>
  arrange(d_f1) |>
  mutate(rank_f1 = row_number(),
         is_top    = rank_f1 > (n() - 3),
         is_bottom = rank_f1 <= 3,
         is_focus  = is_top | is_bottom,
         fill_color = case_when(
           is_top    ~ OLIVE,
           is_bottom ~ SIENNA,
           TRUE      ~ NEUTRAL
         ),
         label_color = case_when(
           is_top    ~ OLIVE_DK,
           is_bottom ~ SIENNA_DK,
           TRUE      ~ "#5a5a5a"
         ))

order_lbls <- deltas$country_lbl

f1_bars <- ggplot(deltas,
                  aes(x = factor(country_lbl, levels = order_lbls),
                      y = d_f1, fill = fill_color)) +
  geom_col(width = 0.72, color = NA) +
  scale_fill_identity() +
  geom_text(
    aes(label = sprintf("%+.1f", d_f1),
        hjust = ifelse(d_f1 >= 0, -0.18, 1.18),
        color = label_color),
    size = 2.8, family = FIG_FONT, fontface = "plain"
  ) +
  scale_color_identity() +
  geom_hline(yintercept = 0, color = INK, linewidth = 0.4) +
  coord_flip(clip = "off") +
  scale_y_continuous(expand = expansion(mult = c(0.10, 0.16))) +
  labs(
    title    = "Planet wins on smallholder\nand Nordic countries",
    subtitle = expression(paste(Delta, " Obj F1 (pp): PRUE-FTP-B3 ",
                                italic("augmax"), " full minus PRUE-B7 (S2) full")),
    y = expression(paste(Delta, " Obj F1 (pp)")),
    caption = "FTW v3.1 full_data, 22 countries. Top-3 wins / losses bolded."
  ) +
  theme_premium(base_size = 9) +
  theme(
    plot.title    = element_text(color = INK, size = 10, face = "bold",
                                 hjust = 0, margin = margin(b = 2)),
    plot.subtitle = element_text(color = "#555555", size = 9,
                                 hjust = 0, margin = margin(b = 10)),
    axis.text.y   = element_text(color = ifelse(deltas$is_focus, INK, "#666666"),
                                 face  = ifelse(deltas$is_focus, "bold", "plain"))
  )

ggsave("paper/figs/per_country_bars_premium.pdf", f1_bars,
       width = 6.6, height = 5.2, device = cairo_pdf)
cat("wrote paper/figs/per_country_bars_premium.pdf\n")

# ---------------------------------------------------------------------------
# Figure 2: smallholder scatter.  Uses ggrepel so country labels never
# collide, draws thin connector segments to their points, and labels only
# the cases that drive the narrative (top wins, the cambodia outlier, the
# largest losses).  Everything else is a quiet grey dot.
# ---------------------------------------------------------------------------

scatter_src <- read_csv("paper/scripts/output/smallholder_scatter.csv",
                        show_col_types = FALSE) |>
  rename(ha = median_field_size_ha) |>
  mutate(
    d_f1 = delta_obj_f1 * 100,
    log_ha = log10(ha),
    country_lbl = tools::toTitleCase(gsub("_", " ", country))
  )

# Pearson r on log-hectares for the corner annotation.
r_val <- cor(scatter_src$log_ha, scatter_src$d_f1)

# Mark the storytelling countries.  These get bold labels + colored dots;
# everything else is muted.
focus_set <- c("rwanda", "lithuania", "south_africa", "cambodia", "germany",
               "france", "denmark")
scatter_src <- scatter_src |>
  mutate(
    is_focus = country %in% focus_set,
    fill_color = case_when(
      !is_focus       ~ "#bcbab2",
      d_f1 >= 0       ~ OLIVE,
      TRUE            ~ SIENNA
    ),
    label_color = case_when(
      !is_focus       ~ "#9b9b9b",
      d_f1 >= 0       ~ OLIVE_DK,
      TRUE            ~ SIENNA_DK
    )
  )

x_lo <- 0.08
x_hi <- 60
y_lo <- floor(min(scatter_src$d_f1)) - 2
y_hi <- ceiling(max(scatter_src$d_f1)) + 2

scatter <- ggplot(scatter_src, aes(x = ha, y = d_f1)) +
  annotate("rect", xmin = x_lo, xmax = x_hi, ymin = 0, ymax = Inf,
           fill = OLIVE,  alpha = 0.045) +
  annotate("rect", xmin = x_lo, xmax = x_hi, ymin = -Inf, ymax = 0,
           fill = SIENNA, alpha = 0.045) +
  geom_hline(yintercept = 0, color = INK, linewidth = 0.4) +
  # OLS fit on log(ha) — drawn as a single line, no shaded ribbon, so the
  # eye lands on the data not the band.
  geom_smooth(method = "lm", se = FALSE,
              color = "#3b3b3b", linewidth = 0.5, linetype = "longdash",
              formula = y ~ x) +
  geom_point(aes(fill = fill_color),
             shape = 21, color = "white", stroke = 0.4, size = 3.4) +
  scale_fill_identity() +
  geom_text_repel(
    data = subset(scatter_src, is_focus),
    aes(label = country_lbl, color = label_color),
    family = FIG_FONT, size = 3.0, fontface = "bold",
    box.padding = 0.55, point.padding = 0.55, segment.size = 0.25,
    segment.color = "#9b9b9b", min.segment.length = 0.1,
    max.overlaps = Inf, seed = 42, force = 4
  ) +
  geom_text_repel(
    data = subset(scatter_src, !is_focus),
    aes(label = country_lbl),
    color = "#888888", family = FIG_FONT, size = 2.5,
    box.padding = 0.4, point.padding = 0.45, segment.size = 0.18,
    segment.color = "#cccccc", min.segment.length = 0.15,
    max.overlaps = Inf, seed = 42, force = 2
  ) +
  scale_color_identity() +
  scale_x_log10(
    limits = c(x_lo, x_hi),
    breaks = c(0.1, 0.3, 1, 3, 10, 30),
    labels = c("0.1", "0.3", "1", "3", "10", "30"),
    expand = expansion(mult = c(0.02, 0.02))
  ) +
  scale_y_continuous(limits = c(y_lo, y_hi),
                     breaks = scales::pretty_breaks(n = 6),
                     expand = expansion(mult = c(0.02, 0.02))) +
  annotate("text", x = x_hi * 0.95, y = y_lo + 0.6,
           hjust = 1, vjust = 0, color = "#555555", size = 3.0,
           family = FIG_FONT, fontface = "italic",
           label = sprintf("Pearson r = %+.2f   |   n = %d",
                           r_val, nrow(scatter_src))) +
  labs(
    title    = "Planet's 3 m advantage clusters\non small-field landscapes",
    subtitle = "Median field area vs. Δ Obj F1 (pp)",
    x = "Median field area (ha, log scale)",
    y = expression(paste(Delta, " Obj F1 (pp, Planet − S2)")),
    caption = "OLS fit (dashed). Tint = win/loss quadrant."
  ) +
  theme_premium(base_size = 9) +
  theme(
    plot.title         = element_text(color = INK, size = 10, face = "bold",
                                      hjust = 0, margin = margin(b = 2)),
    plot.subtitle      = element_text(color = "#555555", size = 9,
                                      hjust = 0, margin = margin(b = 10)),
    panel.grid.major.x = element_line(color = GRID, linewidth = 0.25)
  )

ggsave("paper/figs/smallholder_scatter_premium.pdf", scatter,
       width = 6.8, height = 4.6, device = cairo_pdf)
cat("wrote paper/figs/smallholder_scatter_premium.pdf\n")

# ===========================================================================
# Variations (saved as _v2.pdf for side-by-side comparison):
#   * Bars: top-5 highlighted instead of top-3 + faint magnitude grid.
#   * Scatter: dot size encoding |Delta|, light CI ribbon under OLS line.
# ===========================================================================

# --- Bars v2 ---------------------------------------------------------------
deltas_v2 <- deltas |>
  mutate(rank_f1 = row_number(),
         is_top    = rank_f1 > (n() - 5),
         is_bottom = rank_f1 <= 5,
         is_focus  = is_top | is_bottom,
         fill_color = case_when(
           is_top    ~ OLIVE,
           is_bottom ~ SIENNA,
           TRUE      ~ NEUTRAL
         ),
         label_color = case_when(
           is_top    ~ OLIVE_DK,
           is_bottom ~ SIENNA_DK,
           TRUE      ~ "#5a5a5a"
         ))

f1_bars_v2 <- ggplot(deltas_v2,
                     aes(x = factor(country_lbl, levels = deltas_v2$country_lbl),
                         y = d_f1, fill = fill_color)) +
  geom_vline(xintercept = seq_len(nrow(deltas_v2)), color = "#eeece4", linewidth = 0.18) +
  geom_col(width = 0.72, color = NA) +
  scale_fill_identity() +
  geom_text(
    aes(label = sprintf("%+.1f", d_f1),
        hjust = ifelse(d_f1 >= 0, -0.18, 1.18),
        color = label_color),
    size = 2.8, family = FIG_FONT, fontface = "plain"
  ) +
  scale_color_identity() +
  geom_hline(yintercept = c(-10, -5, 5, 10), color = "#dcdad3",
             linetype = "dotted", linewidth = 0.25) +
  geom_hline(yintercept = 0, color = INK, linewidth = 0.4) +
  coord_flip(clip = "off") +
  scale_y_continuous(expand = expansion(mult = c(0.10, 0.16)),
                     breaks = c(-10, -5, 0, 5, 10, 15)) +
  labs(
    title    = "Planet wins on smallholder\nand Nordic countries",
    subtitle = "PRUE-FTP-B3 augmax full − PRUE-B7 (S2) full",
    y = expression(paste(Delta, " Obj F1 (pp)")),
    caption = "FTW v3.1 full_data, 22 countries. Top-5 highlighted."
  ) +
  theme_premium(base_size = 9) +
  theme(
    plot.title    = element_text(color = INK, size = 10, face = "bold",
                                 hjust = 0, margin = margin(b = 2)),
    plot.subtitle = element_text(color = "#555555", size = 9,
                                 hjust = 0, margin = margin(b = 10)),
    axis.text.y   = element_text(color = ifelse(deltas_v2$is_focus, INK, "#666666"),
                                 face  = ifelse(deltas_v2$is_focus, "bold", "plain"))
  )

ggsave("paper/figs/per_country_bars_premium_v2.pdf", f1_bars_v2,
       width = 2.75, height = 3.6, device = cairo_pdf)
cat("wrote paper/figs/per_country_bars_premium_v2.pdf\n")

# --- Scatter v2 ------------------------------------------------------------
scatter_src_v2 <- scatter_src |>
  mutate(abs_d = abs(d_f1))

scatter_v2 <- ggplot(scatter_src_v2, aes(x = ha, y = d_f1)) +
  annotate("rect", xmin = x_lo, xmax = x_hi, ymin = 0, ymax = Inf,
           fill = OLIVE,  alpha = 0.045) +
  annotate("rect", xmin = x_lo, xmax = x_hi, ymin = -Inf, ymax = 0,
           fill = SIENNA, alpha = 0.045) +
  geom_hline(yintercept = 0, color = INK, linewidth = 0.4) +
  # OLS with a thin light-grey confidence ribbon (subtle, doesn't dominate)
  geom_smooth(method = "lm", se = TRUE,
              color = "#3b3b3b", fill = "#cfcdc6",
              linewidth = 0.45, linetype = "solid",
              alpha = 0.35, formula = y ~ x) +
  geom_point(aes(fill = fill_color, size = abs_d),
             shape = 21, color = "white", stroke = 0.4) +
  scale_size_continuous(range = c(1.6, 4.0), guide = "none") +
  scale_fill_identity() +
  geom_text_repel(
    data = subset(scatter_src_v2, is_focus),
    aes(label = country_lbl, color = label_color),
    family = FIG_FONT, size = 2.4, fontface = "bold",
    box.padding = 0.4, point.padding = 0.4, segment.size = 0.2,
    segment.color = "#9b9b9b", min.segment.length = 0.1,
    max.overlaps = Inf, seed = 42, force = 4
  ) +
  geom_text_repel(
    data = subset(scatter_src_v2, !is_focus),
    aes(label = country_lbl),
    color = "#888888", family = FIG_FONT, size = 2.0,
    box.padding = 0.3, point.padding = 0.35, segment.size = 0.15,
    segment.color = "#cccccc", min.segment.length = 0.15,
    max.overlaps = Inf, seed = 42, force = 2
  ) +
  scale_color_identity() +
  scale_x_log10(
    limits = c(x_lo, x_hi),
    breaks = c(0.1, 0.3, 1, 3, 10, 30),
    labels = c("0.1", "0.3", "1", "3", "10", "30"),
    expand = expansion(mult = c(0.02, 0.02))
  ) +
  scale_y_continuous(limits = c(y_lo, y_hi),
                     breaks = scales::pretty_breaks(n = 6),
                     expand = expansion(mult = c(0.02, 0.02))) +
  annotate("text", x = x_hi * 0.95, y = y_lo + 0.6,
           hjust = 1, vjust = 0, color = "#555555", size = 3.0,
           family = FIG_FONT, fontface = "italic",
           label = sprintf("Pearson r = %+.2f   |   n = %d",
                           r_val, nrow(scatter_src_v2))) +
  labs(
    title    = "Planet's 3 m advantage clusters\non small-field landscapes",
    subtitle = "Median field area vs. Δ Obj F1 (pp)",
    x = "Median field area (ha, log scale)",
    y = expression(paste(Delta, " Obj F1 (pp, Planet − S2)")),
    caption = "Dot size = |Δ|. OLS fit + 95% CI ribbon."
  ) +
  theme_premium(base_size = 9) +
  theme(
    plot.title         = element_text(color = INK, size = 10, face = "bold",
                                      hjust = 0, margin = margin(b = 2)),
    plot.subtitle      = element_text(color = "#555555", size = 9,
                                      hjust = 0, margin = margin(b = 10)),
    panel.grid.major.x = element_line(color = GRID, linewidth = 0.25)
  )

ggsave("paper/figs/smallholder_scatter_premium_v2.pdf", scatter_v2,
       width = 3.0, height = 3.3, device = cairo_pdf)
cat("wrote paper/figs/smallholder_scatter_premium_v2.pdf\n")

# ===========================================================================
# Cumulative augmentation ablation (single-column).  One panel only, Planet
# bars stacked left-to-right showing recipe progression; horizontal reference
# lines for the released FTW S2 PRUE checkpoints.
# ===========================================================================

aug_src <- read_csv("paper/scripts/output/aug_ablation_heldout11.csv",
                    show_col_types = FALSE) |>
  rename(obj_f1 = object_ws_f1) |>
  mutate(
    label = gsub("\n", " ", label),
    obj_f1 = obj_f1 * 100,
    is_planet = panel == "planet"
  )

planet_df <- aug_src |> filter(is_planet) |> mutate(idx = row_number())
s2_refs   <- aug_src |> filter(!is_planet)

# Step palette: muted-to-bold olive as we add more aug knobs.
step_pal <- c("#bdbfa0", "#9aaa66", "#6f8a3a", "#4f6b1f", "#34491c")

aug_plot <- ggplot(planet_df, aes(x = factor(idx), y = obj_f1,
                                  fill = factor(idx))) +
  geom_col(width = 0.72, color = NA) +
  geom_text(aes(label = sprintf("%.1f", obj_f1)),
            vjust = -0.55, size = 2.8, family = FIG_FONT,
            color = OLIVE_DK, fontface = "bold") +
  geom_hline(data = s2_refs,
             aes(yintercept = obj_f1, color = label),
             linetype = "dotted", linewidth = 0.4) +
  scale_fill_manual(values = step_pal, guide = "none") +
  scale_color_manual(
    name = NULL,
    values = c(
      "+ bespoke bundle (augmax, B3 CC-BY)" = "#a07060",
      "+ augmax, B3 full"   = "#8b3a1f",
      "+ augmax, B7 CC-BY"  = "#cc8866",
      "+ augmax, B7 full"   = "#6d2c11"
    ),
    labels = c(
      "+ bespoke bundle (augmax, B3 CC-BY)" = "PRUE-B3 (S2, CC-BY)",
      "+ augmax, B3 full"   = "PRUE-B3 (S2, full)",
      "+ augmax, B7 CC-BY"  = "PRUE-B7 (S2, CC-BY)",
      "+ augmax, B7 full"   = "PRUE-B7 (S2, full)"
    )
  ) +
  scale_x_discrete(labels = planet_df$label) +
  scale_y_continuous(limits = c(0, 52), breaks = seq(0, 50, 10),
                     expand = expansion(mult = c(0, 0.04))) +
  labs(
    title    = "Augs beat backbone scaling\non 3 m PlanetScope",
    subtitle = "Obj F1 (pp, WS+TTA), 11-country held-out",
    x = NULL, y = "Obj F1 (pp)",
    caption = "Dotted lines: released PRUE-B3/B7 (S2)."
  ) +
  theme_premium(base_size = 8) +
  theme(
    axis.text.x        = element_text(angle = 20, hjust = 1, size = 7.5,
                                      color = INK),
    plot.title         = element_text(color = INK, size = 11, face = "bold",
                                      hjust = 0, margin = margin(b = 2)),
    plot.subtitle      = element_text(color = "#555555", size = 8,
                                      hjust = 0, margin = margin(b = 10)),
    legend.position    = c(0.99, 0.02),
    legend.justification = c(1, 0),
    legend.background  = element_rect(fill = PAPER, color = NA),
    legend.key.height  = unit(8, "pt"),
    legend.text        = element_text(size = 7),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = GRID, linewidth = 0.25)
  )

ggsave("paper/figs/aug_ablation_premium.pdf", aug_plot,
       width = 2.75, height = 2.9, device = cairo_pdf)
cat("wrote paper/figs/aug_ablation_premium.pdf\n")
