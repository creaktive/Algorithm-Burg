package Algorithm::Burg;
# ABSTRACT: extrapolate time series using Burg's method

=head1 SYNOPSIS

    #!/usr/bin/env perl;
    use strict;
    use warnings qw(all);
    use Algorithm::Burg;
    ...

=head1 DESCRIPTION

...

=cut

use strict;
use warnings qw(all);

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

...

=cut

has coefficients    => (is => 'rw', isa => ArrayRef[Num]);

=attr order

...

=cut

has order           => (is => 'ro', isa => PositiveInt, required => 1);

=method train($time_series)

...

=cut

sub train {
    my ($self, $time_series) = @_;

    my $m = $self->order;
    my @x = @$time_series;

    # initialize Ak
    my @Ak = (1.0, (0.0) x $m);

    # initialize f and b
    my @f = @$time_series;
    my @b = @$time_series;

    # Initialize Dk
    my $Dk = sum map {
        2.0 * $f[$_] ** 2
    } 0 .. $#f;
    $Dk -= $f[0] ** 2 + $b[$#x] ** 2;

    # Burg recursion
    for my $k (0 .. $m - 1) {
        # compute mu
        my $mu = sum map {
            $f[$_ + $k + 1] * $b[$_]
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
            my $t1 = $f[$n + $k + 1] + $mu * $b[$n];
            my $t2 = $b[$n] + $mu * $f[$n + $k + 1];
            $f[$n + $k + 1] = $t1;
            $b[$n] = $t2;
        }

        # update Dk
        $Dk = (1.0 - $mu ** 2) * $Dk
            - $f[$k + 1] ** 2
            - $b[$#x - $k - 1] ** 2;
    }

    $self->coefficients([ @Ak[1 .. $#Ak] ]);
}

=head1 REFERENCES

=for :list
* L<Burg's Method, Algorithm and Recursion|http://www.emptyloop.com/technotes/A%20tutorial%20on%20Burg's%20method,%20algorithm%20and%20recursion.pdf>
* L<C++ implementation|https://github.com/RhysU/ar/blob/master/collomb2009.cpp>
* L<Matlab/Octave implementation|https://gist.github.com/tobin/2843661>
* L<Python implementation|https://github.com/MrKriss/Old-PhD-Code/blob/master/Algorithms/burg_AR.py>

=cut

1;
