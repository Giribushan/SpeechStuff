##
# BUILD : docker build --no-cache -t sinatra-unicorn-app .
# RUN   : docker run -ti -p 80:8080 sinatra-unicorn-app
##

FROM ruby:2.2-onbuild
MAINTAINER Shayne Parmelee "shayne.parmelee@shopify.com"

# Install app
WORKDIR /usr/src/app
ADD Gemfile /usr/src/Gemfile
ADD Gemfile.lock /usr/src/Gemfile.lock
RUN bundle install --without development test
RUN mkdir tmp
ADD . /usr/src/app

# Clean up APT and bundler when done.
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Run app
CMD bundle exec unicorn -c /usr/src/app/config/unicorn.rb
