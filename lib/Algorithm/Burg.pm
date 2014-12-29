package Algorithm::Burg;
# ABSTRACT: extrapolate time series using Burg's method

=head1 SYNOPSIS

=for test_synopsis
    my (@time_series);

    #!/usr/bin/env perl;
    use strict;
    use warnings qw(all);
    use Algorithm::Burg;
    ...;
    my $burg = Algorithm::Burg->new(order => 150);
    $burg->train(\@time_series);
    my $result = $burg->predict();

=head1 DESCRIPTION

The L<Algorithm::Burg> module uses the Burg method to fit an autoregressive (AR)
model to the input data by minimizing (least squares) the forward and backward
prediction errors while constraining the AR parameters to satisfy the
Levinson-Durbin recursion.

=cut

use strict;
use warnings qw(all);

use Carp qw(croak);
use List::Util qw(sum);

use Moo;
use MooX::Types::MooseLike::Base qw(
    ArrayRef
    Num
);
use MooX::Types::MooseLike::Numeric qw(
    PositiveInt
);

# VERSION

=attr coefficients

AR model polynomial coefficients computed by the C<train> method.

=cut

has coefficients    => (is => 'rwp', isa => ArrayRef[Num]);

=attr order

AR model order

=cut

has order           => (is => 'ro', isa => PositiveInt, required => 1);

=attr series_tail

Store the last L</order> terms of the time series for L</predict($n)>.

=cut

has series_tail     => (is => 'rwp', isa => ArrayRef[Num]);

=method train($time_series)

Computes vector of coefficients using Burg algorithm applied to the input
source data C<$time_series>.

=cut

sub train {
    my ($self, $time_series) = @_;

    croak '$time_series should be an ArrayRef'
        if ref($time_series) ne 'ARRAY';

    my $m = $self->order;
    my @x = @$time_series;

    croak '$time_series should have more elements than the AR order is'
        if $#x < $m;

    # initialize Ak
    my @Ak = (1.0, (0.0) x $m);

    # initialize f and b
    my @f = @$time_series;
    my @B = @$time_series;

    # Initialize Dk
    my $Dk = sum map {
        2.0 * $f[$_] ** 2
    } 0 .. $#f;
    $Dk -= $f[0] ** 2 + $B[$#x] ** 2;

    # Burg recursion
    for my $k (0 .. $m - 1) {
        # compute mu
        my $mu = sum map {
            $f[$_ + $k + 1] * $B[$_]
        } 0 .. $#x - $k - 1;
        $mu *= -2.0 / $Dk;

        # update Ak
        for my $n (0 .. ($k + 1) / 2) {
            my $t1 = $Ak[$n] + $mu * $Ak[$k + 1 - $n];
            my $t2 = $Ak[$k + 1 - $n] + $mu * $Ak[$n];
            $Ak[$n] = $t1;
            $Ak[$k + 1 - $n] = $t2;
        }

        # update f and b
        for my $n (0 .. $#x - $k - 1) {
            my $t1 = $f[$n + $k + 1] + $mu * $B[$n];
            my $t2 = $B[$n] + $mu * $f[$n + $k + 1];
            $f[$n + $k + 1] = $t1;
            $B[$n] = $t2;
        }

        # update Dk
        $Dk = (1.0 - $mu ** 2) * $Dk
            - $f[$k + 1] ** 2
            - $B[$#x - $k - 1] ** 2;
    }

    $self->_set_series_tail([ @x[$#x - $m .. $#x] ]);
    return $self->_set_coefficients([ @Ak[1 .. $#Ak] ]);
}

=method predict($n)

Predict C<$n> next values for the time series. If C<$n> is 0 or bigger than
L</order>, assume C<$n> = L</order>.

=cut

sub predict {
    my ($self, $n) = @_;

    my $coeffs = $self->coefficients;
    my $m = $self->order;
    $n ||= $m
        if !$n || $n > $m;

    my @predicted = @{ $self->series_tail };
    for my $i ($m .. $m + $n) {
        $predicted[$i] = -1.0 * sum map {
            $coeffs->[$_] * $predicted[$i - 1 - $_]
        } 0 .. $m - 1;
    }

    return [ @predicted[$m .. $#predicted] ];
}

=head1 REFERENCES

=for :list
* L<Burg's Method, Algorithm and Recursion|http://www.emptyloop.com/technotes/A%20tutorial%20on%20Burg's%20method,%20algorithm%20and%20recursion.pdf>
* L<C++ implementation|https://github.com/RhysU/ar/blob/master/collomb2009.cpp>
* L<Matlab/Octave implementation|https://gist.github.com/tobin/2843661>
* L<Python implementation|https://github.com/MrKriss/Old-PhD-Code/blob/master/Algorithms/burg_AR.py>

=cut

1;
